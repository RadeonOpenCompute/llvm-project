//===- OMPAllocatableArrayOpt.cpp
//-------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to filter out functions intended for the host
// when compiling for the device and vice versa.
//
//===----------------------------------------------------------------------===//
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

namespace fir {
#define GEN_PASS_DEF_OMPALLOCATABLEARRAYOPT
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {

class OMPAllocatableArrayOptPass
    : public fir::impl::OMPAllocatableArrayOptBase<OMPAllocatableArrayOptPass> {
  struct ArrayBound {
    Value lowerBound;
    Value upperBound;
  };
  struct AllocatableArrayDescriptorItems {
    omp::MapInfoOp basePtrMapInfo;
    Value basePtrKernelArg;
    llvm::SmallVector<ArrayBound> boundDesc;
  };
  llvm::DenseMap<fir::DeclareOp, AllocatableArrayDescriptorItems>
      declareDescriptorMap;

  std::optional<size_t> getNumberOfArrayDim(fir::ArrayCoorOp arrayCoorOp) {
    // TODO: Can we optimize such fir.coor_arr ops?
    if (!arrayCoorOp.getShape()) {
      return {};
    }
    fir::ShapeShiftOp shapeShiftOp =
        dyn_cast<fir::ShapeShiftOp>(arrayCoorOp.getShape().getDefiningOp());
    if (!shapeShiftOp)
      return {};
    return shapeShiftOp.getExtents().size();
  }

  fir::DeclareOp findAllocatableDeclareOp(fir::ArrayCoorOp arrayCoorOp) {
    // TODO: Can we optimize such fir.coor_arr ops?
    if (!arrayCoorOp.getShape()) {
      return nullptr;
    }
    fir::ShapeShiftOp shapeShiftOp =
        dyn_cast<fir::ShapeShiftOp>(arrayCoorOp.getShape().getDefiningOp());
    if (!shapeShiftOp)
      return nullptr;
    fir::BoxAddrOp boxAddrOp =
        dyn_cast<fir::BoxAddrOp>(arrayCoorOp.getMemref().getDefiningOp());
    if (!boxAddrOp)
      return nullptr;
    if (!isa<fir::HeapType>(boxAddrOp.getType()))
      return nullptr;
    fir::LoadOp loadOp =
        dyn_cast<fir::LoadOp>(boxAddrOp.getVal().getDefiningOp());
    if (!loadOp)
      return nullptr;
    return dyn_cast<fir::DeclareOp>(loadOp.getMemref().getDefiningOp());
  }
  void eraseNotUsed(fir::ArrayCoorOp arrayCoorOp,
                    omp::MapClauseOwningOpInterface mapClauseOwner,
                    Block *targetEntryBlock) {
    fir::ShapeShiftOp shapeShiftOp =
        dyn_cast<fir::ShapeShiftOp>(arrayCoorOp.getShape().getDefiningOp());
    fir::BoxAddrOp boxAddrOp =
        dyn_cast<fir::BoxAddrOp>(arrayCoorOp.getMemref().getDefiningOp());
    assert(arrayCoorOp->use_empty());
    arrayCoorOp.erase();
    std::vector<Value> shapeShiftOpExtents(shapeShiftOp.getExtents());
    if (shapeShiftOp->use_empty())
      shapeShiftOp.erase();
    for (size_t i = 0; i < shapeShiftOpExtents.size(); ++i) {
      Value shapeVal = shapeShiftOpExtents[i];
      if (shapeVal.use_empty())
        shapeVal.getDefiningOp()->erase();
    }

    fir::LoadOp loadOp =
        dyn_cast<fir::LoadOp>(boxAddrOp.getVal().getDefiningOp());
    if (boxAddrOp->use_empty())
      boxAddrOp.erase();

    fir::DeclareOp declareOp =
        dyn_cast<fir::DeclareOp>(loadOp.getMemref().getDefiningOp());
    if (loadOp->use_empty())
      loadOp.erase();
    OperandRange mapVarsArr = mapClauseOwner.getMapVarsMutable();
    assert(mapVarsArr.size() == targetEntryBlock->getNumArguments());
    for (size_t i = 0; i < targetEntryBlock->getNumArguments(); ++i) {
      if (targetEntryBlock->getArgument(i) == declareOp.getMemref()) {
        omp::MapInfoOp mapInfo =
            dyn_cast<omp::MapInfoOp>(mapVarsArr[i].getDefiningOp());
        if (declareOp->use_empty()) {
          declareOp.erase();
          targetEntryBlock->eraseArgument(i);
          mapClauseOwner.getMapVarsMutable().erase(i);
          mapInfo.erase();
        }
        break;
      }
    }
  }

  AllocatableArrayDescriptorItems getAllocatableArrayDescriptorItems(
      fir::DeclareOp declareOp, omp::MapClauseOwningOpInterface mapClauseOwner,
      Block *targetEntryBlock, size_t numberOfDims) {
    AllocatableArrayDescriptorItems descriptorItems;
    OperandRange mapVarsArr = mapClauseOwner.getMapVars();
    assert(mapVarsArr.size() == targetEntryBlock->getNumArguments());
    Operation *mapItemVarPtr;
    for (size_t i = 0; i < targetEntryBlock->getNumArguments(); ++i) {
      if (targetEntryBlock->getArgument(i) == declareOp.getMemref()) {
        omp::MapInfoOp mapInfo =
            dyn_cast<omp::MapInfoOp>(mapVarsArr[i].getDefiningOp());
        mapItemVarPtr = mapInfo.getVarPtr().getDefiningOp();
        assert(mapInfo && (mapInfo.getMembers().size() == 1) &&
               "Expected only base addr ptr");
        descriptorItems.basePtrMapInfo =
            dyn_cast<omp::MapInfoOp>(mapInfo.getMembers()[0].getDefiningOp());
        break;
      }
    }
    for (size_t index = 0; index < mapVarsArr.size(); ++index) {
      if (descriptorItems.basePtrMapInfo == mapVarsArr[index].getDefiningOp()) {
        descriptorItems.basePtrKernelArg = targetEntryBlock->getArgument(index);
      }
    }
    assert(descriptorItems.basePtrMapInfo && "Expected base ptr map info");
    assert(descriptorItems.basePtrKernelArg &&
           "Expected base ptr kernel argument");
    return descriptorItems;
  }

  void rewriteMapInfo(AllocatableArrayDescriptorItems &descriptorItem,
                      omp::MapClauseOwningOpInterface mapClauseOwner,
                      Block *targetEntryBlock, size_t numberOfDims) {
    OperandRange mapVarsArr = mapClauseOwner.getMapVars();
    omp::MapInfoOp mapInfo = descriptorItem.basePtrMapInfo;
    size_t index;
    for (index = 0; index < mapVarsArr.size(); ++index) {
      if (descriptorItem.basePtrMapInfo == mapVarsArr[index].getDefiningOp()) {
        break;
      }
    }
    assert(mapInfo);
    OpBuilder opBuilder(mapInfo);
    fir::FirOpBuilder builder(opBuilder, mapInfo);
    Operation *op = opBuilder.create<omp::MapInfoOp>(
        mapInfo->getLoc(), mapInfo.getType(), mapInfo.getVarPtrPtr(),
        TypeAttr::get(mapInfo.getVarType()), mapInfo.getVarPtrPtr(),
        llvm::SmallVector<Value>{}, ArrayAttr{},
        llvm::SmallVector<Value>(mapInfo.getBounds()),
        opBuilder.getIntegerAttr(opBuilder.getIntegerType(64, false),
                                 mapInfo.getMapType().value()),
        opBuilder.getAttr<omp::VariableCaptureKindAttr>(
            mapInfo.getMapCaptureType().value()),
        opBuilder.getStringAttr(""), opBuilder.getBoolAttr(false));

    mapInfo.replaceAllUsesWith(op);
    mapVarsArr[index] = op->getResult(0);

    OpBuilder::InsertPoint insPt = builder.saveInsertionPoint();
    Block *allocaBlock = builder.getAllocaBlock();
    assert(allocaBlock && "No alloca block found for this top level op");
    llvm::SmallVector<Value> newMapOps;
    for (size_t i = 0; i < mapVarsArr.size(); ++i) {
      newMapOps.push_back(mapVarsArr[i]);
    }
    size_t mapArgumentIndex = mapVarsArr.size();
    for (size_t dim = 0; dim < numberOfDims; ++dim) {
      descriptorItem.boundDesc.push_back({});
      for (size_t i = 0; i < 2; ++i) {
        builder.setInsertionPointToStart(allocaBlock);
        auto alloca = builder.create<fir::AllocaOp>(mapInfo->getLoc(),
                                                    builder.getIntegerType(64));
        builder.restoreInsertionPoint(insPt);
        auto dimVal = builder.createIntegerConstant(
            mapInfo->getLoc(), builder.getIndexType(), dim);
        Value allocatableDescriptor =
            builder.create<fir::LoadOp>(mapInfo->getLoc(), mapInfo.getVarPtr());
        auto dimInfo = builder.create<fir::BoxDimsOp>(
            mapInfo->getLoc(), builder.getIndexType(), builder.getIndexType(),
            builder.getIndexType(), allocatableDescriptor, dimVal);
        Value bound =
            builder.createConvert(mapInfo->getLoc(), builder.getIntegerType(64),
                                  dimInfo->getResult(i));
        opBuilder.create<fir::StoreOp>(mapInfo->getLoc(), bound, alloca);
        omp::VariableCaptureKind captureKind = omp::VariableCaptureKind::ByCopy;

        Operation *newMapItem = opBuilder.create<omp::MapInfoOp>(
            mapInfo->getLoc(), alloca.getType(), alloca, TypeAttr::get(bound.getType()),
            Value{}, llvm::SmallVector<Value>{}, ArrayAttr{},
            llvm::SmallVector<Value>{},
            opBuilder.getIntegerAttr(
                opBuilder.getIntegerType(64, false),
                llvm::to_underlying(
                    llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT)),
            builder.getAttr<omp::VariableCaptureKindAttr>(captureKind),
            opBuilder.getStringAttr(""), opBuilder.getBoolAttr(false));

        newMapOps.push_back(newMapItem->getResult(0));
        targetEntryBlock->insertArgument(mapArgumentIndex,
                                         newMapItem->getResult(0).getType(),
                                         newMapItem->getLoc());
        if (i == 0) {
          descriptorItem.boundDesc[dim].lowerBound =
              targetEntryBlock->getArgument(mapArgumentIndex);
        } else if (i == 1) {
          descriptorItem.boundDesc[dim].upperBound =
              targetEntryBlock->getArgument(mapArgumentIndex);
        }
        mapArgumentIndex++;
      }
    }

    mapClauseOwner.getMapVarsMutable().assign(newMapOps);
    mapInfo.erase();
  }

  void rewriteArrayCoorOp(fir::ArrayCoorOp arrayCoorOp,
                          AllocatableArrayDescriptorItems &descriptorItem) {
    OpBuilder opBuilder(arrayCoorOp);
    fir::FirOpBuilder builder(opBuilder, arrayCoorOp);
    Value addr = builder.createConvert(arrayCoorOp.getLoc(),
                                       arrayCoorOp.getMemref().getType(),
                                       descriptorItem.basePtrKernelArg);
    llvm::SmallVector<Value> lbounds;
    llvm::SmallVector<Value> ubounds;

    for (size_t dim = 0; dim < descriptorItem.boundDesc.size(); dim++) {
#if 0
      //Experiment - provide bound information in compile time
      Value lb = descriptorItem.boundDesc[dim].lowerBound;
      lbounds.push_back(builder.createIntegerConstant(lb.getLoc(),builder.getIndexType(), 1));
      ubounds.push_back(builder.createIntegerConstant(lb.getLoc(),builder.getIndexType(), 100));
#else
      Value lb = descriptorItem.boundDesc[dim].lowerBound;
      Value lbVal = builder.create<fir::LoadOp>(lb.getLoc(), lb);
      Value lbValConvert =
          builder.createConvert(lb.getLoc(), builder.getIndexType(), lbVal);
      lbounds.push_back(lbValConvert);
      Value ub = descriptorItem.boundDesc[dim].upperBound;
      Value ubVal = builder.create<fir::LoadOp>(ub.getLoc(), ub);
      Value ubValConvert =
          builder.createConvert(ub.getLoc(), builder.getIndexType(), ubVal);
      ubounds.push_back(ubValConvert);
#endif
    }

    auto shapeShiftArgs = flatZip(lbounds, ubounds);
    auto shapeTy =
        fir::ShapeShiftType::get(arrayCoorOp->getContext(), lbounds.size());
    Value shapeShift = builder.create<fir::ShapeShiftOp>(
        arrayCoorOp.getLoc(), shapeTy, shapeShiftArgs);
    Value optimizedArrayCoorOp = builder.create<fir::ArrayCoorOp>(
        arrayCoorOp.getLoc(), arrayCoorOp.getType(), addr, shapeShift,
        arrayCoorOp.getSlice(), arrayCoorOp.getIndices(),
        arrayCoorOp.getTypeparams());
    arrayCoorOp.replaceAllUsesWith(optimizedArrayCoorOp);
  }

public:
  OMPAllocatableArrayOptPass() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    declareDescriptorMap.clear();
    func->walk<WalkOrder::PreOrder>([&](omp::TargetOp targetOp) {
      auto mapClauseOwner = llvm::dyn_cast<omp::MapClauseOwningOpInterface>(
          targetOp.getOperation());
      Block *entryBlock = &targetOp->getRegion(0).front();
      if (mapClauseOwner) {
        OperandRange mapVarsArr = mapClauseOwner.getMapVars();
        targetOp->walk<WalkOrder::PreOrder>([&](fir::ArrayCoorOp arrayCoorOp) {
          fir::DeclareOp declareOp = findAllocatableDeclareOp(arrayCoorOp);
          std::optional<size_t> numberOfArrayDim =
              getNumberOfArrayDim(arrayCoorOp);
          if (!numberOfArrayDim.has_value())
            return;
          if (!declareOp)
            return;
          if (!declareDescriptorMap.contains(declareOp)) {
            declareDescriptorMap[declareOp] =
                getAllocatableArrayDescriptorItems(
                    declareOp, mapClauseOwner, entryBlock, *numberOfArrayDim);
            rewriteMapInfo(declareDescriptorMap[declareOp], mapClauseOwner,
                           entryBlock, *numberOfArrayDim);
          }
          rewriteArrayCoorOp(arrayCoorOp, declareDescriptorMap[declareOp]);
          eraseNotUsed(arrayCoorOp, mapClauseOwner, entryBlock);
        });
        ;
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> fir::createOMPAllocatableArrayOptPass() {
  return std::make_unique<OMPAllocatableArrayOptPass>();
}
