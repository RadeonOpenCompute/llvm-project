//===-- OpenMP.cpp -- Open MP directive lowering --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/OpenMP.h"

#include "ClauseProcessor.h"
#include "DataSharingProcessor.h"
#include "DirectivesCommon.h"
#include "ReductionProcessor.h"
#include "flang/Common/idioms.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/openmp-directive-sets.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using namespace Fortran::lower::omp;

//===----------------------------------------------------------------------===//
// HostClausesInsertionGuard
//===----------------------------------------------------------------------===//

/// If the insertion point of the builder is located inside of an omp.target
/// region, this RAII guard moves the insertion point to just before that
/// omp.target operation and then restores the original insertion point when
/// destroyed. If not currently inserting inside an omp.target, it remains
/// unchanged.
class HostClausesInsertionGuard {
public:
  HostClausesInsertionGuard(mlir::OpBuilder &builder) : builder(builder) {
    targetOp = findParentTargetOp(builder);
    if (targetOp) {
      ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(targetOp);
    }
  }

  ~HostClausesInsertionGuard() {
    if (ip.isSet()) {
      fixupExtractedHostOps();
      builder.restoreInsertionPoint(ip);
    }
  }

private:
  mlir::OpBuilder &builder;
  mlir::OpBuilder::InsertPoint ip;
  mlir::omp::TargetOp targetOp;

  /// Fixup any uses of target region block arguments that we have just created
  /// outside of the target region, and replace them by their host values.
  void fixupExtractedHostOps() {
    auto useOutsideTargetRegion = [](mlir::OpOperand &operand) {
      if (mlir::Operation *owner = operand.getOwner())
        return !owner->getParentOfType<mlir::omp::TargetOp>();
      return false;
    };

    mlir::OperandRange map = targetOp.getMapOperands();
    for (mlir::BlockArgument arg : targetOp.getRegion().getArguments()) {
      mlir::Value hostVal = map[arg.getArgNumber()]
                                .getDefiningOp<mlir::omp::MapInfoOp>()
                                .getVarPtr();

      // Replace instances of omp.target block arguments used outside with their
      // corresponding host value.
      arg.replaceUsesWithIf(hostVal, [&](mlir::OpOperand &operand) -> bool {
        // If the use is an hlfir.declare, we need to search for the matching
        // one within host code.
        if (auto declareOp = llvm::dyn_cast_if_present<hlfir::DeclareOp>(
                operand.getOwner())) {
          if (auto hostDeclareOp = hostVal.getDefiningOp<hlfir::DeclareOp>()) {
            declareOp->replaceUsesWithIf(hostDeclareOp.getResults(),
                                         useOutsideTargetRegion);
          } else if (auto hostBoxOp = hostVal.getDefiningOp<fir::BoxAddrOp>()) {
            declareOp->replaceUsesWithIf(hostBoxOp.getVal()
                                             .getDefiningOp<hlfir::DeclareOp>()
                                             .getResults(),
                                         useOutsideTargetRegion);
          }
        }
        return useOutsideTargetRegion(operand);
      });
    }
  }
};

//===----------------------------------------------------------------------===//
// OpWithBodyGenInfo
//===----------------------------------------------------------------------===//

struct OpWithBodyGenInfo {
  /// A type for a code-gen callback function. This takes as argument the op for
  /// which the code is being generated and returns the arguments of the op's
  /// region.
  using GenOMPRegionEntryCBFn =
      std::function<llvm::SmallVector<const Fortran::semantics::Symbol *>(
          mlir::Operation *)>;

  OpWithBodyGenInfo(Fortran::lower::AbstractConverter &converter,
                    Fortran::semantics::SemanticsContext &semaCtx,
                    mlir::Location loc, Fortran::lower::pft::Evaluation &eval)
      : converter(converter), semaCtx(semaCtx), loc(loc), eval(eval) {}

  OpWithBodyGenInfo &setGenNested(bool value) {
    genNested = value;
    return *this;
  }

  OpWithBodyGenInfo &setOuterCombined(bool value) {
    outerCombined = value;
    return *this;
  }

  OpWithBodyGenInfo &setClauses(const Fortran::parser::OmpClauseList *value) {
    clauses = value;
    return *this;
  }

  OpWithBodyGenInfo &setDataSharingProcessor(DataSharingProcessor *value) {
    dsp = value;
    return *this;
  }

  OpWithBodyGenInfo &
  setReductions(llvm::ArrayRef<const Fortran::semantics::Symbol *> symbols,
                llvm::ArrayRef<mlir::Type> types) {
    reductionSymbols = symbols;
    reductionTypes = types;
    return *this;
  }

  OpWithBodyGenInfo &setGenRegionEntryCb(GenOMPRegionEntryCBFn value) {
    genRegionEntryCB = value;
    return *this;
  }

  /// [inout] converter to use for the clauses.
  Fortran::lower::AbstractConverter &converter;
  /// [in] Semantics context
  Fortran::semantics::SemanticsContext &semaCtx;
  /// [in] location in source code.
  mlir::Location loc;
  /// [in] current PFT node/evaluation.
  Fortran::lower::pft::Evaluation &eval;
  /// [in] whether to generate FIR for nested evaluations
  bool genNested = true;
  /// [in] is this an outer operation - prevents privatization.
  bool outerCombined = false;
  /// [in] list of clauses to process.
  const Fortran::parser::OmpClauseList *clauses = nullptr;
  /// [in] if provided, processes the construct's data-sharing attributes.
  DataSharingProcessor *dsp = nullptr;
  /// [in] if provided, list of reduction symbols
  llvm::ArrayRef<const Fortran::semantics::Symbol *> reductionSymbols;
  /// [in] if provided, list of reduction types
  llvm::ArrayRef<mlir::Type> reductionTypes;
  /// [in] if provided, emits the op's region entry. Otherwise, an emtpy block
  /// is created in the region.
  GenOMPRegionEntryCBFn genRegionEntryCB = nullptr;
};

//===----------------------------------------------------------------------===//
// Code generation helper functions
//===----------------------------------------------------------------------===//

static Fortran::lower::pft::Evaluation *
getCollapsedLoopEval(Fortran::lower::pft::Evaluation &eval, int collapseValue) {
  // Return the Evaluation of the innermost collapsed loop, or the current one
  // if there was no COLLAPSE.
  if (collapseValue == 0)
    return &eval;

  Fortran::lower::pft::Evaluation *curEval = &eval.getFirstNestedEvaluation();
  for (int i = 1; i < collapseValue; i++) {
    // The nested evaluations should be DoConstructs (i.e. they should form
    // a loop nest). Each DoConstruct is a tuple <NonLabelDoStmt, Block,
    // EndDoStmt>.
    assert(curEval->isA<Fortran::parser::DoConstruct>());
    curEval = &*std::next(curEval->getNestedEvaluations().begin());
  }
  return curEval;
}

static void genNestedEvaluations(Fortran::lower::AbstractConverter &converter,
                                 Fortran::lower::pft::Evaluation &eval,
                                 int collapseValue = 0) {
  Fortran::lower::pft::Evaluation *curEval =
      getCollapsedLoopEval(eval, collapseValue);

  for (Fortran::lower::pft::Evaluation &e : curEval->getNestedEvaluations())
    converter.genEval(e);
}

static fir::GlobalOp globalInitialization(
    Fortran::lower::AbstractConverter &converter,
    fir::FirOpBuilder &firOpBuilder, const Fortran::semantics::Symbol &sym,
    const Fortran::lower::pft::Variable &var, mlir::Location currentLocation) {
  mlir::Type ty = converter.genType(sym);
  std::string globalName = converter.mangleName(sym);
  mlir::StringAttr linkage = firOpBuilder.createInternalLinkage();
  fir::GlobalOp global =
      firOpBuilder.createGlobal(currentLocation, ty, globalName, linkage);

  // Create default initialization for non-character scalar.
  if (Fortran::semantics::IsAllocatableOrObjectPointer(&sym)) {
    mlir::Type baseAddrType = ty.dyn_cast<fir::BoxType>().getEleTy();
    Fortran::lower::createGlobalInitialization(
        firOpBuilder, global, [&](fir::FirOpBuilder &b) {
          mlir::Value nullAddr =
              b.createNullConstant(currentLocation, baseAddrType);
          mlir::Value box =
              b.create<fir::EmboxOp>(currentLocation, ty, nullAddr);
          b.create<fir::HasValueOp>(currentLocation, box);
        });
  } else {
    Fortran::lower::createGlobalInitialization(
        firOpBuilder, global, [&](fir::FirOpBuilder &b) {
          mlir::Value undef = b.create<fir::UndefOp>(currentLocation, ty);
          b.create<fir::HasValueOp>(currentLocation, undef);
        });
  }

  return global;
}

static mlir::Operation *getCompareFromReductionOp(mlir::Operation *reductionOp,
                                                  mlir::Value loadVal) {
  for (mlir::Value reductionOperand : reductionOp->getOperands()) {
    if (mlir::Operation *compareOp = reductionOperand.getDefiningOp()) {
      if (compareOp->getOperand(0) == loadVal ||
          compareOp->getOperand(1) == loadVal)
        assert((mlir::isa<mlir::arith::CmpIOp>(compareOp) ||
                mlir::isa<mlir::arith::CmpFOp>(compareOp)) &&
               "Expected comparison not found in reduction intrinsic");
      return compareOp;
    }
  }
  return nullptr;
}

// Get the extended value for \p val by extracting additional variable
// information from \p base.
static fir::ExtendedValue getExtendedValue(fir::ExtendedValue base,
                                           mlir::Value val) {
  return base.match(
      [&](const fir::MutableBoxValue &box) -> fir::ExtendedValue {
        return fir::MutableBoxValue(val, box.nonDeferredLenParams(), {});
      },
      [&](const auto &) -> fir::ExtendedValue {
        return fir::substBase(base, val);
      });
}

#ifndef NDEBUG
static bool isThreadPrivate(Fortran::lower::SymbolRef sym) {
  if (const auto *details =
          sym->detailsIf<Fortran::semantics::CommonBlockDetails>()) {
    for (const auto &obj : details->objects())
      if (!obj->test(Fortran::semantics::Symbol::Flag::OmpThreadprivate))
        return false;
    return true;
  }
  return sym->test(Fortran::semantics::Symbol::Flag::OmpThreadprivate);
}
#endif

static void threadPrivatizeVars(Fortran::lower::AbstractConverter &converter,
                                Fortran::lower::pft::Evaluation &eval) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  // If the symbol corresponds to the original ThreadprivateOp, use the symbol
  // value from that operation to create one ThreadprivateOp copy operation
  // inside the parallel region.
  // In some cases, however, the symbol will correspond to the original,
  // non-threadprivate variable. This can happen, for instance, with a common
  // block, declared in a separate module, used by a parent procedure and
  // privatized in its child procedure.
  auto genThreadprivateOp = [&](Fortran::lower::SymbolRef sym) -> mlir::Value {
    assert(isThreadPrivate(sym));
    mlir::Value symValue = converter.getSymbolAddress(sym);
    mlir::Operation *op = symValue.getDefiningOp();
    if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(op))
      op = declOp.getMemref().getDefiningOp();
    if (mlir::isa<mlir::omp::ThreadprivateOp>(op))
      symValue = mlir::dyn_cast<mlir::omp::ThreadprivateOp>(op).getSymAddr();
    return firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  };

  llvm::SetVector<const Fortran::semantics::Symbol *> threadprivateSyms;
  converter.collectSymbolSet(
      eval, threadprivateSyms,
      Fortran::semantics::Symbol::Flag::OmpThreadprivate);
  std::set<Fortran::semantics::SourceName> threadprivateSymNames;

  // For a COMMON block, the ThreadprivateOp is generated for itself instead of
  // its members, so only bind the value of the new copied ThreadprivateOp
  // inside the parallel region to the common block symbol only once for
  // multiple members in one COMMON block.
  llvm::SetVector<const Fortran::semantics::Symbol *> commonSyms;
  for (std::size_t i = 0; i < threadprivateSyms.size(); i++) {
    const Fortran::semantics::Symbol *sym = threadprivateSyms[i];
    mlir::Value symThreadprivateValue;
    // The variable may be used more than once, and each reference has one
    // symbol with the same name. Only do once for references of one variable.
    if (threadprivateSymNames.find(sym->name()) != threadprivateSymNames.end())
      continue;
    threadprivateSymNames.insert(sym->name());
    if (const Fortran::semantics::Symbol *common =
            Fortran::semantics::FindCommonBlockContaining(sym->GetUltimate())) {
      mlir::Value commonThreadprivateValue;
      if (commonSyms.contains(common)) {
        commonThreadprivateValue = converter.getSymbolAddress(*common);
      } else {
        commonThreadprivateValue = genThreadprivateOp(*common);
        converter.bindSymbol(*common, commonThreadprivateValue);
        commonSyms.insert(common);
      }
      symThreadprivateValue = Fortran::lower::genCommonBlockMember(
          converter, currentLocation, *sym, commonThreadprivateValue);
    } else {
      symThreadprivateValue = genThreadprivateOp(*sym);
    }

    fir::ExtendedValue sexv = converter.getSymbolExtendedValue(*sym);
    fir::ExtendedValue symThreadprivateExv =
        getExtendedValue(sexv, symThreadprivateValue);
    converter.bindSymbol(*sym, symThreadprivateExv);
  }

  firOpBuilder.restoreInsertionPoint(insPt);
}

/// Create the body (block) for an OpenMP Operation.
///
/// \param [in]   op - the operation the body belongs to.
/// \param [in] info - options controlling code-gen for the construction.
template <typename Op>
static void createBodyOfOp(Op &op, const OpWithBodyGenInfo &info) {
  fir::FirOpBuilder &firOpBuilder = info.converter.getFirOpBuilder();

  auto insertMarker = [](fir::FirOpBuilder &builder) {
    mlir::Value undef = builder.create<fir::UndefOp>(builder.getUnknownLoc(),
                                                     builder.getIndexType());
    return undef.getDefiningOp();
  };

  // If an argument for the region is provided then create the block with that
  // argument. Also update the symbol's address with the mlir argument value.
  // e.g. For loops the argument is the induction variable. And all further
  // uses of the induction variable should use this mlir value.
  auto regionArgs =
      [&]() -> llvm::SmallVector<const Fortran::semantics::Symbol *> {
    if (info.genRegionEntryCB != nullptr) {
      return info.genRegionEntryCB(op);
    }

    firOpBuilder.createBlock(&op.getRegion());
    return {};
  }();
  // Mark the earliest insertion point.
  mlir::Operation *marker = insertMarker(firOpBuilder);

  // If it is an unstructured region and is not the outer region of a combined
  // construct, create empty blocks for all evaluations.
  if (info.eval.lowerAsUnstructured() && !info.outerCombined)
    Fortran::lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp,
                                            mlir::omp::YieldOp>(
        firOpBuilder, info.eval.getNestedEvaluations());

  // Start with privatization, so that the lowering of the nested
  // code will use the right symbols.
  // TODO Check that nothing broke from replacing WsLoopOp and SimdLoopOp here.
  constexpr bool isLoop = std::is_same_v<Op, mlir::omp::LoopNestOp>;
  bool privatize = info.clauses && !info.outerCombined;

  firOpBuilder.setInsertionPoint(marker);
  std::optional<DataSharingProcessor> tempDsp;
  if (privatize) {
    if (!info.dsp) {
      tempDsp.emplace(info.converter, *info.clauses, info.eval);
      tempDsp->processStep1();
    }
  }

  if constexpr (std::is_same_v<Op, mlir::omp::ParallelOp>) {
    threadPrivatizeVars(info.converter, info.eval);
    if (info.clauses) {
      firOpBuilder.setInsertionPoint(marker);
      CopyinClauseOps clauseOps;
      ClauseProcessor(info.converter, info.semaCtx, *info.clauses)
          .processCopyin(clauseOps);
    }
  }

  if (info.genNested) {
    // genFIR(Evaluation&) tries to patch up unterminated blocks, causing
    // a lot of complications for our approach if the terminator generation
    // is delayed past this point. Insert a temporary terminator here, then
    // delete it.
    firOpBuilder.setInsertionPointToEnd(&op.getRegion().back());
    auto *temp = Fortran::lower::genOpenMPTerminator(
        firOpBuilder, op.getOperation(), info.loc);
    firOpBuilder.setInsertionPointAfter(marker);
    genNestedEvaluations(info.converter, info.eval);
    temp->erase();
  }

  // Get or create a unique exiting block from the given region, or
  // return nullptr if there is no exiting block.
  auto getUniqueExit = [&](mlir::Region &region) -> mlir::Block * {
    // Find the blocks where the OMP terminator should go. In simple cases
    // it is the single block in the operation's region. When the region
    // is more complicated, especially with unstructured control flow, there
    // may be multiple blocks, and some of them may have non-OMP terminators
    // resulting from lowering of the code contained within the operation.
    // All the remaining blocks are potential exit points from the op's region.
    //
    // Explicit control flow cannot exit any OpenMP region (other than via
    // STOP), and that is enforced by semantic checks prior to lowering. STOP
    // statements are lowered to a function call.

    // Collect unterminated blocks.
    llvm::SmallVector<mlir::Block *> exits;
    for (mlir::Block &b : region) {
      if (b.empty() || !b.back().hasTrait<mlir::OpTrait::IsTerminator>())
        exits.push_back(&b);
    }

    if (exits.empty())
      return nullptr;
    // If there already is a unique exiting block, do not create another one.
    // Additionally, some ops (e.g. omp.sections) require only 1 block in
    // its region.
    if (exits.size() == 1)
      return exits[0];
    mlir::Block *exit = firOpBuilder.createBlock(&region);
    for (mlir::Block *b : exits) {
      firOpBuilder.setInsertionPointToEnd(b);
      firOpBuilder.create<mlir::cf::BranchOp>(info.loc, exit);
    }
    return exit;
  };

  if (auto *exitBlock = getUniqueExit(op.getRegion())) {
    firOpBuilder.setInsertionPointToEnd(exitBlock);
    auto *term = Fortran::lower::genOpenMPTerminator(
        firOpBuilder, op.getOperation(), info.loc);
    // Only insert lastprivate code when there actually is an exit block.
    // Such a block may not exist if the nested code produced an infinite
    // loop (this may not make sense in production code, but a user could
    // write that and we should handle it).
    firOpBuilder.setInsertionPoint(term);
    if (privatize) {
      if (!info.dsp) {
        assert(tempDsp.has_value());
        tempDsp->processStep2(op, isLoop);
      } else {
        if (isLoop && regionArgs.size() > 0)
          info.dsp->setLoopIV(info.converter.getSymbolAddress(*regionArgs[0]));
        info.dsp->processStep2(op, isLoop);
      }
    }
  }

  firOpBuilder.setInsertionPointAfter(marker);
  marker->erase();
}

static void genBodyOfTargetDataOp(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval, bool genNested,
    mlir::omp::DataOp &dataOp,
    const llvm::SmallVector<mlir::Type> &useDeviceTypes,
    const llvm::SmallVector<mlir::Location> &useDeviceLocs,
    const llvm::SmallVector<const Fortran::semantics::Symbol *>
        &useDeviceSymbols,
    const mlir::Location &currentLocation) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Region &region = dataOp.getRegion();

  firOpBuilder.createBlock(&region, {}, useDeviceTypes, useDeviceLocs);

  for (auto [argIndex, argSymbol] : llvm::enumerate(useDeviceSymbols)) {
    const mlir::BlockArgument &arg = region.front().getArgument(argIndex);
    fir::ExtendedValue extVal = converter.getSymbolExtendedValue(*argSymbol);
    if (auto refType = arg.getType().dyn_cast<fir::ReferenceType>()) {
      if (fir::isa_builtin_cptr_type(refType.getElementType())) {
        converter.bindSymbol(*argSymbol, arg);
      } else {
        // Avoid capture of a reference to a structured binding.
        const Fortran::semantics::Symbol *sym = argSymbol;
        extVal.match(
            [&](const fir::MutableBoxValue &mbv) {
              converter.bindSymbol(
                  *sym,
                  fir::MutableBoxValue(
                      arg, fir::factory::getNonDeferredLenParams(extVal), {}));
            },
            [&](const auto &) {
              TODO(converter.getCurrentLocation(),
                   "use_device clause operand unsupported type");
            });
      }
    } else {
      TODO(converter.getCurrentLocation(),
           "use_device clause operand unsupported type");
    }
  }

  // Insert dummy instruction to remember the insertion position. The
  // marker will be deleted by clean up passes since there are no uses.
  // Remembering the position for further insertion is important since
  // there are hlfir.declares inserted above while setting block arguments
  // and new code from the body should be inserted after that.
  mlir::Value undefMarker = firOpBuilder.create<fir::UndefOp>(
      dataOp.getOperation()->getLoc(), firOpBuilder.getIndexType());

  // Create blocks for unstructured regions. This has to be done since
  // blocks are initially allocated with the function as the parent region.
  if (eval.lowerAsUnstructured()) {
    Fortran::lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp,
                                            mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());
  }

  firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);

  // Set the insertion point after the marker.
  firOpBuilder.setInsertionPointAfter(undefMarker.getDefiningOp());
  if (genNested)
    genNestedEvaluations(converter, eval);
}

// This functions creates a block for the body of the targetOp's region. It adds
// all the symbols present in mapSymbols as block arguments to this block.
static void genBodyOfTargetOp(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval, bool genNested,
    mlir::omp::TargetOp &targetOp,
    const llvm::SmallVector<mlir::Type> &mapSymTypes,
    const llvm::SmallVector<mlir::Location> &mapSymLocs,
    const llvm::SmallVector<const Fortran::semantics::Symbol *> &mapSymbols,
    const mlir::Location &currentLocation) {
  assert(mapSymTypes.size() == mapSymLocs.size());

  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Region &region = targetOp.getRegion();

  auto *regionBlock =
      firOpBuilder.createBlock(&region, {}, mapSymTypes, mapSymLocs);

  // Clones the `bounds` placing them inside the target region and returns them.
  auto cloneBound = [&](mlir::Value bound) {
    if (mlir::isMemoryEffectFree(bound.getDefiningOp())) {
      mlir::Operation *clonedOp = bound.getDefiningOp()->clone();
      regionBlock->push_back(clonedOp);
      return clonedOp->getResult(0);

    }
    TODO(converter.getCurrentLocation(),
         "target map clause operand unsupported bound type");
  };

  auto cloneBounds = [cloneBound](llvm::ArrayRef<mlir::Value> bounds) {
    llvm::SmallVector<mlir::Value> clonedBounds;
    for (mlir::Value bound : bounds)
      clonedBounds.emplace_back(cloneBound(bound));
    return clonedBounds;
  };

  // Bind the symbols to their corresponding block arguments.
  for (auto [argIndex, argSymbol] : llvm::enumerate(mapSymbols)) {
    const mlir::BlockArgument &arg = region.getArgument(argIndex);
    // Avoid capture of a reference to a structured binding.
    const Fortran::semantics::Symbol *sym = argSymbol;
    // Structure component symbols don't have bindings.
    if (sym->owner().IsDerivedType())
      continue;
    fir::ExtendedValue extVal = converter.getSymbolExtendedValue(*sym);
    extVal.match(
        [&](const fir::BoxValue &v) {
          converter.bindSymbol(*sym,
                               fir::BoxValue(arg, cloneBounds(v.getLBounds()),
                                             v.getExplicitParameters(),
                                             v.getExplicitExtents()));
        },
        [&](const fir::MutableBoxValue &v) {
          converter.bindSymbol(
              *sym, fir::MutableBoxValue(arg, cloneBounds(v.getLBounds()),
                                         v.getMutableProperties()));
        },
        [&](const fir::ArrayBoxValue &v) {
          converter.bindSymbol(
              *sym, fir::ArrayBoxValue(arg, cloneBounds(v.getExtents()),
                                       cloneBounds(v.getLBounds()),
                                       v.getSourceBox()));
        },
        [&](const fir::CharArrayBoxValue &v) {
          converter.bindSymbol(
              *sym, fir::CharArrayBoxValue(arg, cloneBound(v.getLen()),
                                           cloneBounds(v.getExtents()),
                                           cloneBounds(v.getLBounds())));
        },
        [&](const fir::CharBoxValue &v) {
          converter.bindSymbol(*sym,
                               fir::CharBoxValue(arg, cloneBound(v.getLen())));
        },
        [&](const fir::UnboxedValue &v) { converter.bindSymbol(*sym, arg); },
        [&](const auto &) {
          TODO(converter.getCurrentLocation(),
               "target map clause operand unsupported type");
        });
  }

  // Check if cloning the bounds introduced any dependency on the outer region.
  // If so, then either clone them as well if they are MemoryEffectFree, or else
  // copy them to a new temporary and add them to the map and block_argument
  // lists and replace their uses with the new temporary.
  llvm::SetVector<mlir::Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  while (!valuesDefinedAbove.empty()) {
    for (mlir::Value val : valuesDefinedAbove) {
      mlir::Operation *valOp = val.getDefiningOp();
      if (mlir::isMemoryEffectFree(valOp)) {
        mlir::Operation *clonedOp = valOp->clone();
        regionBlock->push_front(clonedOp);
        val.replaceUsesWithIf(
            clonedOp->getResult(0), [regionBlock](mlir::OpOperand &use) {
              return use.getOwner()->getBlock() == regionBlock;
            });
      } else {
        auto savedIP = firOpBuilder.getInsertionPoint();
        firOpBuilder.setInsertionPointAfter(valOp);
        auto copyVal =
            firOpBuilder.createTemporary(val.getLoc(), val.getType());
        firOpBuilder.createStoreWithConvert(copyVal.getLoc(), val, copyVal);

        llvm::SmallVector<mlir::Value> bounds;
        std::stringstream name;
        firOpBuilder.setInsertionPoint(targetOp);
        mlir::Value mapOp = createMapInfoOp(
            firOpBuilder, copyVal.getLoc(), copyVal, mlir::Value{}, name.str(),
            bounds, llvm::SmallVector<mlir::Value>{},
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT),
            mlir::omp::VariableCaptureKind::ByCopy, copyVal.getType());
        targetOp.getMapOperandsMutable().append(mapOp);
        mlir::Value clonedValArg =
            region.addArgument(copyVal.getType(), copyVal.getLoc());
        firOpBuilder.setInsertionPointToStart(regionBlock);
        auto loadOp = firOpBuilder.create<fir::LoadOp>(clonedValArg.getLoc(),
                                                       clonedValArg);
        val.replaceUsesWithIf(
            loadOp->getResult(0), [regionBlock](mlir::OpOperand &use) {
              return use.getOwner()->getBlock() == regionBlock;
            });
        firOpBuilder.setInsertionPoint(regionBlock, savedIP);
      }
    }
    valuesDefinedAbove.clear();
    mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  }

  // Insert dummy instruction to remember the insertion position. The
  // marker will be deleted since there are not uses.
  // In the HLFIR flow there are hlfir.declares inserted above while
  // setting block arguments.
  mlir::Value undefMarker = firOpBuilder.create<fir::UndefOp>(
      targetOp.getOperation()->getLoc(), firOpBuilder.getIndexType());

  // Create blocks for unstructured regions. This has to be done since
  // blocks are initially allocated with the function as the parent region.
  if (eval.lowerAsUnstructured()) {
    Fortran::lower::createEmptyRegionBlocks<mlir::omp::TerminatorOp,
                                            mlir::omp::YieldOp>(
        firOpBuilder, eval.getNestedEvaluations());
  }

  firOpBuilder.create<mlir::omp::TerminatorOp>(currentLocation);

  // Create the insertion point after the marker.
  firOpBuilder.setInsertionPointAfter(undefMarker.getDefiningOp());
  if (genNested)
    genNestedEvaluations(converter, eval);
}

template <typename OpTy, typename... Args>
static OpTy genOpWithBody(const OpWithBodyGenInfo &info, Args &&...args) {
  auto op = info.converter.getFirOpBuilder().create<OpTy>(
      info.loc, std::forward<Args>(args)...);
  createBodyOfOp<OpTy>(op, info);
  return op;
}

static mlir::Value
calculateTripCount(Fortran::lower::AbstractConverter &converter,
                   mlir::Location loc, llvm::ArrayRef<mlir::Value> lbs,
                   llvm::ArrayRef<mlir::Value> ubs,
                   llvm::ArrayRef<mlir::Value> steps) {
  using namespace mlir::arith;
  assert(lbs.size() == ubs.size() && lbs.size() == steps.size() &&
         !lbs.empty() && "Invalid bounds or step");

  fir::FirOpBuilder &b = converter.getFirOpBuilder();

  // Get the bit width of an integer-like type.
  auto widthOf = [](mlir::Type ty) -> unsigned {
    if (mlir::isa<mlir::IndexType>(ty)) {
      return mlir::IndexType::kInternalStorageBitWidth;
    }
    if (auto tyInt = mlir::dyn_cast<mlir::IntegerType>(ty)) {
      return tyInt.getWidth();
    }
    llvm_unreachable("Unexpected type");
  };

  // For a type that is either IntegerType or IndexType, return the
  // equivalent IntegerType. In the former case this is a no-op.
  auto asIntTy = [&](mlir::Type ty) -> mlir::IntegerType {
    if (ty.isIndex()) {
      return mlir::IntegerType::get(ty.getContext(), widthOf(ty));
    }
    assert(ty.isIntOrIndex() && "Unexpected type");
    return mlir::cast<mlir::IntegerType>(ty);
  };

  // For two given values, establish a common signless IntegerType
  // that can represent any value of type of x and of type of y,
  // and return the pair of x, y converted to the new type.
  auto unifyToSignless =
      [&](fir::FirOpBuilder &b, mlir::Value x,
          mlir::Value y) -> std::pair<mlir::Value, mlir::Value> {
    auto tyX = asIntTy(x.getType()), tyY = asIntTy(y.getType());
    unsigned width = std::max(widthOf(tyX), widthOf(tyY));
    auto wideTy = mlir::IntegerType::get(b.getContext(), width,
                                         mlir::IntegerType::Signless);
    return std::make_pair(b.createConvert(loc, wideTy, x),
                          b.createConvert(loc, wideTy, y));
  };

  // Start with signless i32 by default.
  auto tripCount = b.createIntegerConstant(loc, b.getI32Type(), 1);

  for (auto [origLb, origUb, origStep] : llvm::zip(lbs, ubs, steps)) {
    auto tmpS0 = b.createIntegerConstant(loc, origStep.getType(), 0);
    auto [step, step0] = unifyToSignless(b, origStep, tmpS0);
    auto reverseCond = b.create<CmpIOp>(loc, CmpIPredicate::slt, step, step0);
    auto negStep = b.create<SubIOp>(loc, step0, step);
    mlir::Value absStep = b.create<SelectOp>(loc, reverseCond, negStep, step);

    auto [lb, ub] = unifyToSignless(b, origLb, origUb);
    auto start = b.create<SelectOp>(loc, reverseCond, ub, lb);
    auto end = b.create<SelectOp>(loc, reverseCond, lb, ub);

    mlir::Value range = b.create<SubIOp>(loc, end, start);
    auto rangeCond = b.create<CmpIOp>(loc, CmpIPredicate::slt, end, start);
    std::tie(range, absStep) = unifyToSignless(b, range, absStep);
    // numSteps = (range /u absStep) + 1
    auto numSteps =
        b.create<AddIOp>(loc, b.create<DivUIOp>(loc, range, absStep),
                         b.createIntegerConstant(loc, range.getType(), 1));

    auto trip0 = b.createIntegerConstant(loc, numSteps.getType(), 0);
    auto loopTripCount = b.create<SelectOp>(loc, rangeCond, trip0, numSteps);
    auto [totalTC, thisTC] = unifyToSignless(b, tripCount, loopTripCount);
    tripCount = b.create<MulIOp>(loc, totalTC, thisTC);
  }

  return tripCount;
}

static bool evalHasSiblings(Fortran::lower::pft::Evaluation &eval) {
  return eval.parent.visit(Fortran::common::visitors{
      [&](const Fortran::lower::pft::Program &parent) {
        return parent.getUnits().size() + parent.getCommonBlocks().size() > 1;
      },
      [&](const Fortran::lower::pft::Evaluation &parent) {
        for (auto &sibling : *parent.evaluationList)
          if (&sibling != &eval && !sibling.isEndStmt())
            return true;

        return false;
      },
      [&](const auto &parent) {
        for (auto &sibling : parent.evaluationList)
          if (&sibling != &eval && !sibling.isEndStmt())
            return true;

        return false;
      }});
}

/// Extract the list of function and variable symbols affected by the given
/// 'declare target' directive and return the intended device type for them.
static mlir::omp::DeclareTargetDeviceType getDeclareTargetInfo(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclareTargetConstruct &declareTargetConstruct,
    EnterLinkToClauseOps &enterLinkToClauseOps) {

  // The default capture type
  DeviceTypeClauseOps deviceTypeClauseOps = {
      mlir::omp::DeclareTargetDeviceType::any};
  const auto &spec = std::get<Fortran::parser::OmpDeclareTargetSpecifier>(
      declareTargetConstruct.t);

  if (const auto *objectList{
          Fortran::parser::Unwrap<Fortran::parser::OmpObjectList>(spec.u)}) {
    // Case: declare target(func, var1, var2)
    gatherFuncAndVarSyms(*objectList, mlir::omp::DeclareTargetCaptureClause::to,
                         enterLinkToClauseOps.symbolAndClause);
  } else if (const auto *clauseList{
                 Fortran::parser::Unwrap<Fortran::parser::OmpClauseList>(
                     spec.u)}) {
    if (clauseList->v.empty()) {
      // Case: declare target, implicit capture of function
      enterLinkToClauseOps.symbolAndClause.emplace_back(
          mlir::omp::DeclareTargetCaptureClause::to,
          eval.getOwningProcedure()->getSubprogramSymbol());
    }

    ClauseProcessor cp(converter, semaCtx, *clauseList);
    cp.processDeviceType(deviceTypeClauseOps);
    cp.processEnter(enterLinkToClauseOps);
    cp.processLink(enterLinkToClauseOps);
    cp.processTo(enterLinkToClauseOps);
    cp.processTODO<Fortran::parser::OmpClause::Indirect>(
        converter.getCurrentLocation(),
        llvm::omp::Directive::OMPD_declare_target);
  }

  return deviceTypeClauseOps.deviceType;
}

static void collectDeferredDeclareTargets(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclareTargetConstruct &declareTargetConstruct,
    llvm::SmallVectorImpl<Fortran::lower::OMPDeferredDeclareTargetInfo>
        &deferredDeclareTarget) {
  EnterLinkToClauseOps clauseOps;
  mlir::omp::DeclareTargetDeviceType devType = getDeclareTargetInfo(
      converter, semaCtx, eval, declareTargetConstruct, clauseOps);
  // Return the device type only if at least one of the targets for the
  // directive is a function or subroutine
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();

  for (const DeclareTargetCapturePair &symClause : clauseOps.symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(converter.mangleName(
        std::get<const Fortran::semantics::Symbol &>(symClause)));

    if (!op) {
      deferredDeclareTarget.push_back(
          {std::get<0>(symClause), devType, std::get<1>(symClause)});
    }
  }
}

static std::optional<mlir::omp::DeclareTargetDeviceType>
getDeclareTargetFunctionDevice(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclareTargetConstruct
        &declareTargetConstruct) {
  EnterLinkToClauseOps clauseOps;
  mlir::omp::DeclareTargetDeviceType deviceType = getDeclareTargetInfo(
      converter, semaCtx, eval, declareTargetConstruct, clauseOps);

  // Return the device type only if at least one of the targets for the
  // directive is a function or subroutine
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  for (const DeclareTargetCapturePair &symClause : clauseOps.symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(converter.mangleName(
        std::get<const Fortran::semantics::Symbol &>(symClause)));

    if (mlir::isa_and_nonnull<mlir::func::FuncOp>(op))
      return deviceType;
  }

  return std::nullopt;
}

static mlir::Operation *
createAndSetPrivatizedLoopVar(Fortran::lower::AbstractConverter &converter,
                              mlir::Location loc, mlir::Value indexVal,
                              const Fortran::semantics::Symbol *sym) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::OpBuilder::InsertPoint insPt = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPointToStart(firOpBuilder.getAllocaBlock());

  mlir::Type tempTy = converter.genType(*sym);
  mlir::Value temp = firOpBuilder.create<fir::AllocaOp>(
      loc, tempTy, /*pinned=*/true, /*lengthParams=*/mlir::ValueRange{},
      /*shapeParams*/ mlir::ValueRange{},
      llvm::ArrayRef<mlir::NamedAttribute>{
          fir::getAdaptToByRefAttr(firOpBuilder)});
  converter.bindSymbol(*sym, temp);
  firOpBuilder.restoreInsertionPoint(insPt);
  mlir::Value cvtVal = firOpBuilder.createConvert(loc, tempTy, indexVal);
  mlir::Operation *storeOp = firOpBuilder.create<fir::StoreOp>(
      loc, cvtVal, converter.getSymbolAddress(*sym));
  return storeOp;
}

static void
genLoopVars(mlir::Operation *op, Fortran::lower::AbstractConverter &converter,
            mlir::Location &loc,
            llvm::ArrayRef<const Fortran::semantics::Symbol *> args) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  auto &region = op->getRegion(0);

  std::size_t loopVarTypeSize = 0;
  for (const Fortran::semantics::Symbol *arg : args)
    loopVarTypeSize = std::max(loopVarTypeSize, arg->GetUltimate().size());
  mlir::Type loopVarType = getLoopVarType(converter, loopVarTypeSize);
  llvm::SmallVector<mlir::Type> tiv(args.size(), loopVarType);
  llvm::SmallVector<mlir::Location> locs(args.size(), loc);
  firOpBuilder.createBlock(&region, {}, tiv, locs);
  // The argument is not currently in memory, so make a temporary for the
  // argument, and store it there, then bind that location to the argument.
  mlir::Operation *storeOp = nullptr;
  for (auto [argIndex, argSymbol] : llvm::enumerate(args)) {
    mlir::Value indexVal = fir::getBase(region.front().getArgument(argIndex));
    storeOp =
        createAndSetPrivatizedLoopVar(converter, loc, indexVal, argSymbol);
  }
  firOpBuilder.setInsertionPointAfter(storeOp);
}

static void genReductionVars(
    mlir::Operation *op, Fortran::lower::AbstractConverter &converter,
    mlir::Location &loc,
    llvm::ArrayRef<const Fortran::semantics::Symbol *> reductionArgs,
    llvm::ArrayRef<mlir::Type> reductionTypes) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  llvm::SmallVector<mlir::Location> blockArgLocs(reductionArgs.size(), loc);

  mlir::Block *entryBlock = firOpBuilder.createBlock(
      &op->getRegion(0), {}, reductionTypes, blockArgLocs);

  // Bind the reduction arguments to their block arguments
  for (auto [arg, prv] :
       llvm::zip_equal(reductionArgs, entryBlock->getArguments())) {
    converter.bindSymbol(*arg, prv);
  }
}

//===----------------------------------------------------------------------===//
// Code generation functions for clauses
//===----------------------------------------------------------------------===//

// TODO Try to compile, check privatization of simple wsloop/simdloop/distribute
// TODO Move common args and functions into a ConstructProcessor class

static void
genCriticalDeclareClauses(Fortran::lower::AbstractConverter &converter,
                          Fortran::semantics::SemanticsContext &semaCtx,
                          const Fortran::parser::OmpClauseList &clauses,
                          mlir::Location loc,
                          CriticalDeclareOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processHint(clauseOps);
}

static void genDataClauses(Fortran::lower::AbstractConverter &converter,
                           Fortran::semantics::SemanticsContext &semaCtx,
                           Fortran::lower::StatementContext &stmtCtx,
                           const Fortran::parser::OmpClauseList &clauses,
                           mlir::Location loc, DataOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processIf(Fortran::parser::OmpIfClause::DirectiveNameModifier::TargetData,
               clauseOps);
  cp.processMap(loc, stmtCtx, clauseOps);
  cp.processUseDeviceAddr(clauseOps);
  cp.processUseDevicePtr(clauseOps);
}

static void genDistributeClauses(Fortran::lower::AbstractConverter &converter,
                                 Fortran::semantics::SemanticsContext &semaCtx,
                                 const Fortran::parser::OmpClauseList &clauses,
                                 mlir::Location loc,
                                 DistributeOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processTODO<Fortran::parser::OmpClause::DistSchedule,
                 Fortran::parser::OmpClause::Order>(
      loc, llvm::omp::Directive::OMPD_distribute);
}

static void genEnterExitUpdateDataClauses(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::StatementContext &stmtCtx,
    const Fortran::parser::OmpClauseList &clauses, mlir::Location loc,
    Fortran::parser::OmpIfClause::DirectiveNameModifier directive,
    EnterExitUpdateDataOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processDepend(clauseOps);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processIf(directive, clauseOps);
  cp.processNowait(clauseOps);

  if (directive ==
      Fortran::parser::OmpIfClause::DirectiveNameModifier::TargetUpdate) {
    cp.processMotionClauses<Fortran::parser::OmpClause::To>(stmtCtx, clauseOps);
    cp.processMotionClauses<Fortran::parser::OmpClause::From>(stmtCtx,
                                                              clauseOps);
  } else {
    cp.processMap(loc, stmtCtx, clauseOps);
  }
}

static void genFlushClauses(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    const std::optional<Fortran::parser::OmpObjectList> &objects,
    const std::optional<std::list<Fortran::parser::OmpMemoryOrderClause>>
        &clauses,
    mlir::Location loc, llvm::SmallVectorImpl<mlir::Value> &operandRange) {
  if (objects)
    genObjectList(*objects, converter, operandRange);

  if (clauses && clauses->size() > 0)
    TODO(converter.getCurrentLocation(), "Handle OmpMemoryOrderClause");
}

static void genLoopNestClauses(Fortran::lower::AbstractConverter &converter,
                               Fortran::semantics::SemanticsContext &semaCtx,
                               Fortran::lower::pft::Evaluation &eval,
                               const Fortran::parser::OmpClauseList &clauses,
                               mlir::Location loc,
                               LoopNestOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processCollapse(loc, eval, clauseOps);
}

static void
genOrderedRegionClauses(Fortran::lower::AbstractConverter &converter,
                        Fortran::semantics::SemanticsContext &semaCtx,
                        const Fortran::parser::OmpClauseList &clauses,
                        mlir::Location loc,
                        OrderedRegionOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processTODO<Fortran::parser::OmpClause::Simd,
                 Fortran::parser::OmpClause::Threads>(
      loc, llvm::omp::Directive::OMPD_ordered);
}

static void genParallelClauses(Fortran::lower::AbstractConverter &converter,
                               Fortran::semantics::SemanticsContext &semaCtx,
                               Fortran::lower::StatementContext &stmtCtx,
                               const Fortran::parser::OmpClauseList &clauses,
                               mlir::Location loc, bool processReduction,
                               bool evalNumThreadsOutsideTarget,
                               ParallelOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processDefault();
  cp.processIf(Fortran::parser::OmpIfClause::DirectiveNameModifier::Parallel,
               clauseOps);
  cp.processProcBind(clauseOps);

  if (processReduction)
    cp.processReduction(loc, clauseOps);

  if (evalNumThreadsOutsideTarget) {
    HostClausesInsertionGuard guard(converter.getFirOpBuilder());
    cp.processNumThreads(stmtCtx, clauseOps);
  } else {
    cp.processNumThreads(stmtCtx, clauseOps);
  }
}

static void genSectionsClauses(Fortran::lower::AbstractConverter &converter,
                               Fortran::semantics::SemanticsContext &semaCtx,
                               const Fortran::parser::OmpClauseList &clauses,
                               mlir::Location loc,
                               bool clausesFromBeginSections,
                               SectionsOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  if (clausesFromBeginSections) {
    cp.processAllocate(clauseOps);
    cp.processSectionsReduction(loc, clauseOps);
  } else {
    cp.processNowait(clauseOps);
  }
}

static void genSimdLoopClauses(Fortran::lower::AbstractConverter &converter,
                               Fortran::semantics::SemanticsContext &semaCtx,
                               Fortran::lower::StatementContext &stmtCtx,
                               const Fortran::parser::OmpClauseList &clauses,
                               mlir::Location loc,
                               SimdLoopOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processIf(Fortran::parser::OmpIfClause::DirectiveNameModifier::Simd,
               clauseOps);
  cp.processReduction(loc, clauseOps);
  cp.processSafelen(clauseOps);
  cp.processSimdlen(clauseOps);
  cp.processTODO<Fortran::parser::OmpClause::Aligned,
                 Fortran::parser::OmpClause::Linear,
                 Fortran::parser::OmpClause::Nontemporal,
                 Fortran::parser::OmpClause::Order>(
      loc, llvm::omp::Directive::OMPD_simd);
}

static void genSingleClauses(Fortran::lower::AbstractConverter &converter,
                             Fortran::semantics::SemanticsContext &semaCtx,
                             const Fortran::parser::OmpClauseList &beginClauses,
                             const Fortran::parser::OmpClauseList &endClauses,
                             mlir::Location loc, SingleOpClauseOps &clauseOps) {
  ClauseProcessor bcp(converter, semaCtx, beginClauses);
  bcp.processAllocate(clauseOps);

  ClauseProcessor ecp(converter, semaCtx, endClauses);
  ecp.processCopyprivate(loc, clauseOps);
  ecp.processNowait(clauseOps);
}

static void genTargetClauses(Fortran::lower::AbstractConverter &converter,
                             Fortran::semantics::SemanticsContext &semaCtx,
                             Fortran::lower::StatementContext &stmtCtx,
                             const Fortran::parser::OmpClauseList &clauses,
                             mlir::Location loc, bool processHostOnlyClauses,
                             bool processReduction,
                             TargetOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processDepend(clauseOps);
  cp.processDevice(stmtCtx, clauseOps);
  cp.processIf(Fortran::parser::OmpIfClause::DirectiveNameModifier::Target,
               clauseOps);
  cp.processMap(loc, stmtCtx, clauseOps);
  cp.processThreadLimit(stmtCtx, clauseOps);

  if (processHostOnlyClauses)
    cp.processNowait(clauseOps);

  if (processReduction)
    cp.processTargetReduction(clauseOps);

  cp.processTODO<Fortran::parser::OmpClause::Allocate,
                 Fortran::parser::OmpClause::Defaultmap,
                 Fortran::parser::OmpClause::Firstprivate,
                 Fortran::parser::OmpClause::HasDeviceAddr,
                 Fortran::parser::OmpClause::InReduction,
                 Fortran::parser::OmpClause::IsDevicePtr,
                 Fortran::parser::OmpClause::Private,
                 Fortran::parser::OmpClause::UsesAllocators>(
      loc, llvm::omp::Directive::OMPD_target);
}

static void genTaskClauses(Fortran::lower::AbstractConverter &converter,
                           Fortran::semantics::SemanticsContext &semaCtx,
                           Fortran::lower::StatementContext &stmtCtx,
                           const Fortran::parser::OmpClauseList &clauses,
                           mlir::Location loc, TaskOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processDefault();
  cp.processDepend(clauseOps);
  cp.processFinal(stmtCtx, clauseOps);
  cp.processIf(Fortran::parser::OmpIfClause::DirectiveNameModifier::Task,
               clauseOps);
  cp.processMergeable(clauseOps);
  cp.processPriority(stmtCtx, clauseOps);
  cp.processUntied(clauseOps);
  cp.processTODO<Fortran::parser::OmpClause::InReduction,
                 Fortran::parser::OmpClause::Detach,
                 Fortran::parser::OmpClause::Affinity>(
      loc, llvm::omp::Directive::OMPD_task);
}

static void genTaskGroupClauses(Fortran::lower::AbstractConverter &converter,
                                Fortran::semantics::SemanticsContext &semaCtx,
                                const Fortran::parser::OmpClauseList &clauses,
                                mlir::Location loc,
                                TaskGroupOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processTODO<Fortran::parser::OmpClause::TaskReduction>(
      loc, llvm::omp::Directive::OMPD_taskgroup);
}

static void genTaskLoopClauses(Fortran::lower::AbstractConverter &converter,
                               Fortran::semantics::SemanticsContext &semaCtx,
                               Fortran::lower::StatementContext &stmtCtx,
                               const Fortran::parser::OmpClauseList &clauses,
                               mlir::Location loc,
                               TaskLoopOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processDefault();
  cp.processFinal(stmtCtx, clauseOps);
  cp.processIf(Fortran::parser::OmpIfClause::DirectiveNameModifier::Taskloop,
               clauseOps);
  cp.processMergeable(clauseOps);
  cp.processPriority(stmtCtx, clauseOps);
  cp.processReduction(loc, clauseOps);
  cp.processUntied(clauseOps);
  cp.processTODO<Fortran::parser::OmpClause::Grainsize,
                 Fortran::parser::OmpClause::InReduction,
                 Fortran::parser::OmpClause::Nogroup,
                 Fortran::parser::OmpClause::NumTasks>(
      loc, llvm::omp::Directive::OMPD_taskloop);
}

static void genTaskWaitClauses(Fortran::lower::AbstractConverter &converter,
                               Fortran::semantics::SemanticsContext &semaCtx,
                               const Fortran::parser::OmpClauseList &clauses,
                               mlir::Location loc,
                               TaskWaitOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processTODO<Fortran::parser::OmpClause::Depend,
                 Fortran::parser::OmpClause::Nowait>(
      loc, llvm::omp::Directive::OMPD_taskwait);
}

static void genTeamsClauses(Fortran::lower::AbstractConverter &converter,
                            Fortran::semantics::SemanticsContext &semaCtx,
                            Fortran::lower::StatementContext &stmtCtx,
                            const Fortran::parser::OmpClauseList &clauses,
                            mlir::Location loc, bool evalNumTeamsOutsideTarget,
                            TeamsOpClauseOps &clauseOps) {
  ClauseProcessor cp(converter, semaCtx, clauses);
  cp.processAllocate(clauseOps);
  cp.processIf(Fortran::parser::OmpIfClause::DirectiveNameModifier::Teams,
               clauseOps);
  cp.processDefault();

  if (evalNumTeamsOutsideTarget) {
    HostClausesInsertionGuard guard(converter.getFirOpBuilder());
    cp.processNumTeams(stmtCtx, clauseOps);
    cp.processThreadLimit(stmtCtx, clauseOps);
  } else {
    cp.processNumTeams(stmtCtx, clauseOps);
    cp.processThreadLimit(stmtCtx, clauseOps);
  }
}

static void genWsLoopClauses(Fortran::lower::AbstractConverter &converter,
                             Fortran::semantics::SemanticsContext &semaCtx,
                             Fortran::lower::StatementContext &stmtCtx,
                             const Fortran::parser::OmpClauseList &beginClauses,
                             const Fortran::parser::OmpClauseList *endClauses,
                             mlir::Location loc, WsloopOpClauseOps &clauseOps) {
  ClauseProcessor bcp(converter, semaCtx, beginClauses);
  bcp.processOrdered(clauseOps);
  bcp.processReduction(loc, clauseOps);
  bcp.processSchedule(stmtCtx, clauseOps);

  if (endClauses) {
    ClauseProcessor ecp(converter, semaCtx, *endClauses);
    ecp.processNowait(clauseOps);
  }

  bcp.processTODO<Fortran::parser::OmpClause::Linear,
                  Fortran::parser::OmpClause::Order>(
      loc, llvm::omp::Directive::OMPD_do);
}

//===----------------------------------------------------------------------===//
// Code generation functions for leaf constructs
//===----------------------------------------------------------------------===//

// TODO Pass <X>OpClauseOps as arg to all gen<X>Op

static mlir::omp::BarrierOp
genBarrierOp(Fortran::lower::AbstractConverter &converter,
             Fortran::semantics::SemanticsContext &semaCtx,
             Fortran::lower::pft::Evaluation &eval,
             mlir::Location currentLocation) {
  return converter.getFirOpBuilder().create<mlir::omp::BarrierOp>(
      currentLocation);
}

static mlir::omp::CriticalOp
genCriticalOp(Fortran::lower::AbstractConverter &converter,
              Fortran::semantics::SemanticsContext &semaCtx,
              Fortran::lower::pft::Evaluation &eval, bool genNested,
              mlir::Location currentLocation,
              const Fortran::parser::OmpClauseList &clauseList,
              const std::optional<Fortran::parser::Name> &name) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::FlatSymbolRefAttr nameAttr;

  if (name.has_value()) {
    CriticalDeclareOpClauseOps clauseOps;
    genCriticalDeclareClauses(converter, semaCtx, clauseList, currentLocation,
                              clauseOps);

    std::string nameStr = name.value().ToString();
    mlir::ModuleOp module = firOpBuilder.getModule();
    auto global = module.lookupSymbol<mlir::omp::CriticalDeclareOp>(nameStr);
    if (!global) {
      mlir::OpBuilder modBuilder(module.getBodyRegion());
      global = modBuilder.create<mlir::omp::CriticalDeclareOp>(
          currentLocation, firOpBuilder.getStringAttr(nameStr),
          clauseOps.hintAttr);
    }
    nameAttr = mlir::FlatSymbolRefAttr::get(firOpBuilder.getContext(),
                                            global.getSymName());
  }

  return genOpWithBody<mlir::omp::CriticalOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested),
      nameAttr);
}

static mlir::omp::DataOp
genDataOp(Fortran::lower::AbstractConverter &converter,
          Fortran::semantics::SemanticsContext &semaCtx,
          Fortran::lower::pft::Evaluation &eval, bool genNested,
          mlir::Location currentLocation,
          const Fortran::parser::OmpClauseList &clauseList) {
  Fortran::lower::StatementContext stmtCtx;
  DataOpClauseOps clauseOps;
  genDataClauses(converter, semaCtx, stmtCtx, clauseList, currentLocation,
                 clauseOps);

  auto dataOp = converter.getFirOpBuilder().create<mlir::omp::DataOp>(
      currentLocation, clauseOps.ifVar, clauseOps.deviceVar,
      clauseOps.useDevicePtrVars, clauseOps.useDeviceAddrVars,
      clauseOps.mapVars);

  genBodyOfTargetDataOp(converter, semaCtx, eval, genNested, dataOp,
                        clauseOps.useDeviceTypes, clauseOps.useDeviceLocs,
                        clauseOps.useDeviceSymbols, currentLocation);
  return dataOp;
}

static mlir::omp::DistributeOp
genDistributeOp(Fortran::lower::AbstractConverter &converter,
                Fortran::semantics::SemanticsContext &semaCtx,
                Fortran::lower::pft::Evaluation &eval, bool isComposite,
                mlir::Location currentLocation,
                const Fortran::parser::OmpClauseList &clauseList,
                bool outerCombined = false) {
  DistributeOpClauseOps clauseOps;
  genDistributeClauses(converter, semaCtx, clauseList, currentLocation,
                       clauseOps);

  return genOpWithBody<mlir::omp::DistributeOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(false)
          .setOuterCombined(outerCombined)
          .setClauses(&clauseList),
      clauseOps.distScheduleStaticAttr, clauseOps.distScheduleChunkSizeVar,
      clauseOps.allocateVars, clauseOps.allocatorVars, clauseOps.orderAttr,
      isComposite ? converter.getFirOpBuilder().getUnitAttr() : nullptr);
}

template <typename OpTy>
static OpTy
genEnterExitUpdateDataOp(Fortran::lower::AbstractConverter &converter,
                         Fortran::semantics::SemanticsContext &semaCtx,
                         mlir::Location currentLocation,
                         const Fortran::parser::OmpClauseList &clauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;

  Fortran::parser::OmpIfClause::DirectiveNameModifier directive;
  if constexpr (std::is_same_v<OpTy, mlir::omp::EnterDataOp>) {
    directive =
        Fortran::parser::OmpIfClause::DirectiveNameModifier::TargetEnterData;
  } else if constexpr (std::is_same_v<OpTy, mlir::omp::ExitDataOp>) {
    directive =
        Fortran::parser::OmpIfClause::DirectiveNameModifier::TargetExitData;
  } else if constexpr (std::is_same_v<OpTy, mlir::omp::UpdateDataOp>) {
    directive =
        Fortran::parser::OmpIfClause::DirectiveNameModifier::TargetUpdate;
  } else {
    llvm_unreachable("Unexpected TARGET data construct");
  }

  EnterExitUpdateDataOpClauseOps clauseOps;
  genEnterExitUpdateDataClauses(converter, semaCtx, stmtCtx, clauseList,
                                currentLocation, directive, clauseOps);

  return firOpBuilder.create<OpTy>(
      currentLocation, clauseOps.ifVar, clauseOps.deviceVar,
      clauseOps.dependTypeAttrs.empty()
          ? nullptr
          : firOpBuilder.getArrayAttr(clauseOps.dependTypeAttrs),
      clauseOps.dependVars, clauseOps.nowaitAttr, clauseOps.mapVars);
}

static mlir::omp::FlushOp
genFlushOp(Fortran::lower::AbstractConverter &converter,
           Fortran::semantics::SemanticsContext &semaCtx,
           Fortran::lower::pft::Evaluation &eval,
           mlir::Location currentLocation,
           const std::optional<Fortran::parser::OmpObjectList> &objectList,
           const std::optional<std::list<Fortran::parser::OmpMemoryOrderClause>>
               &clauseList) {
  llvm::SmallVector<mlir::Value, 4> operandRange;
  genFlushClauses(converter, semaCtx, objectList, clauseList, currentLocation,
                  operandRange);

  return converter.getFirOpBuilder().create<mlir::omp::FlushOp>(
      converter.getCurrentLocation(), operandRange);
}

static mlir::omp::LoopNestOp
genLoopNestOp(Fortran::lower::AbstractConverter &converter,
              Fortran::semantics::SemanticsContext &semaCtx,
              Fortran::lower::pft::Evaluation &eval,
              mlir::Location currentLocation,
              const Fortran::parser::OmpClauseList &clauseList,
              const LoopNestOpClauseOps &clauseOps, DataSharingProcessor &dsp) {
  auto *nestedEval =
      getCollapsedLoopEval(eval, Fortran::lower::getCollapseValue(clauseList));

  auto ivCallback = [&](mlir::Operation *op) {
    genLoopVars(op, converter, currentLocation, clauseOps.loopIV);
    return clauseOps.loopIV;
  };

  return genOpWithBody<mlir::omp::LoopNestOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, *nestedEval)
          .setClauses(&clauseList)
          .setDataSharingProcessor(&dsp)
          .setGenRegionEntryCb(ivCallback)
          .setGenNested(true),
      clauseOps.loopLBVar, clauseOps.loopUBVar, clauseOps.loopStepVar,
      /*inclusive=*/converter.getFirOpBuilder().getUnitAttr());
}

static mlir::omp::MasterOp
genMasterOp(Fortran::lower::AbstractConverter &converter,
            Fortran::semantics::SemanticsContext &semaCtx,
            Fortran::lower::pft::Evaluation &eval, bool genNested,
            mlir::Location currentLocation) {
  return genOpWithBody<mlir::omp::MasterOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested),
      /*resultTypes=*/mlir::TypeRange());
}

static mlir::omp::OrderedRegionOp
genOrderedRegionOp(Fortran::lower::AbstractConverter &converter,
                   Fortran::semantics::SemanticsContext &semaCtx,
                   Fortran::lower::pft::Evaluation &eval, bool genNested,
                   mlir::Location currentLocation,
                   const Fortran::parser::OmpClauseList &clauseList) {
  OrderedRegionOpClauseOps clauseOps;
  genOrderedRegionClauses(converter, semaCtx, clauseList, currentLocation,
                          clauseOps);

  // TODO Store clauseOps.parLevelThreadsAttr in op.
  return genOpWithBody<mlir::omp::OrderedRegionOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested),
      clauseOps.parLevelSimdAttr);
}

static mlir::omp::ParallelOp
genParallelOp(Fortran::lower::AbstractConverter &converter,
              Fortran::lower::SymMap &symTable,
              Fortran::semantics::SemanticsContext &semaCtx,
              Fortran::lower::pft::Evaluation &eval, bool genNested,
              bool isComposite, mlir::Location currentLocation,
              const Fortran::parser::OmpClauseList &clauseList,
              bool outerCombined = false) {
  // TODO Distinguish between genParallelOp as block vs wrapper
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  ParallelOpClauseOps clauseOps;
  clauseOps.reductionSymbols.emplace();

  auto offloadModOp =
      llvm::cast<mlir::omp::OffloadModuleInterface>(*converter.getModuleOp());
  mlir::omp::TargetOp targetOp = findParentTargetOp(firOpBuilder);

  bool evalNumThreadsOutsideTarget =
      targetOp && !offloadModOp.getIsTargetDevice() && !evalHasSiblings(eval);

  genParallelClauses(converter, semaCtx, stmtCtx, clauseList, currentLocation,
                     /*processReduction=*/!outerCombined,
                     evalNumThreadsOutsideTarget, clauseOps);

  auto reductionCallback = [&](mlir::Operation *op) {
    genReductionVars(op, converter, currentLocation,
                     *clauseOps.reductionSymbols, clauseOps.reductionTypes);
    return *clauseOps.reductionSymbols;
  };

  OpWithBodyGenInfo genInfo =
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setOuterCombined(outerCombined)
          .setClauses(&clauseList)
          .setReductions(*clauseOps.reductionSymbols, clauseOps.reductionTypes)
          .setGenRegionEntryCb(reductionCallback);

  if (!enableDelayedPrivatization) {
    auto parallelOp = genOpWithBody<mlir::omp::ParallelOp>(
        genInfo, clauseOps.ifVar, /*num_threads_var=*/nullptr,
        clauseOps.allocateVars, clauseOps.allocatorVars,
        clauseOps.reductionVars,
        clauseOps.reductionDeclSymbols.empty()
            ? nullptr
            : firOpBuilder.getArrayAttr(clauseOps.reductionDeclSymbols),
        clauseOps.procBindKindAttr, clauseOps.privateVars,
        clauseOps.privatizers.empty()
            ? nullptr
            : firOpBuilder.getArrayAttr(clauseOps.privatizers),
        isComposite ? firOpBuilder.getUnitAttr() : nullptr);

    if (clauseOps.numThreadsVar) {
      if (evalNumThreadsOutsideTarget)
        targetOp.getNumThreadsMutable().assign(clauseOps.numThreadsVar);
      else
        parallelOp.getNumThreadsVarMutable().assign(clauseOps.numThreadsVar);
    }

    return parallelOp;
  }

  // TODO Integrate delayed privatization better with the new approach.
  //   - Store delayedPrivatizationInfo.{originalAddresses,privatizers} in
  // clauseOps.{privateVars,privatizers}.
  //   - Outline genRegionEntryCB into composable genPrivatizedVars.
  //   - Refactor to create the omp.parallel op in a single place and possibly
  // only use a single callback.
  //   - Check whether the external DataSharingProcessor could be used, and skip
  // the call to processStep1() here. Perhaps also skip setting it in the
  // OpWithBodyGenInfo structure.

  bool privatize = !outerCombined;
  DataSharingProcessor dsp(converter, clauseList, eval,
                           /*useDelayedPrivatization=*/true, &symTable);

  if (privatize)
    dsp.processStep1();

  const auto &delayedPrivatizationInfo = dsp.getDelayedPrivatizationInfo();

  auto genRegionEntryCB = [&](mlir::Operation *op) {
    auto parallelOp = llvm::cast<mlir::omp::ParallelOp>(op);

    llvm::SmallVector<mlir::Location> reductionLocs(
        clauseOps.reductionVars.size(), currentLocation);

    mlir::OperandRange privateVars = parallelOp.getPrivateVars();
    mlir::Region &region = parallelOp.getRegion();

    llvm::SmallVector<mlir::Type> privateVarTypes = clauseOps.reductionTypes;
    privateVarTypes.reserve(privateVarTypes.size() + privateVars.size());
    llvm::transform(privateVars, std::back_inserter(privateVarTypes),
                    [](mlir::Value v) { return v.getType(); });

    llvm::SmallVector<mlir::Location> privateVarLocs = reductionLocs;
    privateVarLocs.reserve(privateVarLocs.size() + privateVars.size());
    llvm::transform(privateVars, std::back_inserter(privateVarLocs),
                    [](mlir::Value v) { return v.getLoc(); });

    converter.getFirOpBuilder().createBlock(&region, /*insertPt=*/{},
                                            privateVarTypes, privateVarLocs);

    llvm::SmallVector<const Fortran::semantics::Symbol *> allSymbols =
        *clauseOps.reductionSymbols;
    allSymbols.append(delayedPrivatizationInfo.symbols);
    for (auto [arg, prv] : llvm::zip_equal(allSymbols, region.getArguments())) {
      converter.bindSymbol(*arg, prv);
    }

    return allSymbols;
  };

  // TODO Merge with the reduction CB.
  genInfo.setGenRegionEntryCb(genRegionEntryCB).setDataSharingProcessor(&dsp);

  llvm::SmallVector<mlir::Attribute> privatizers(
      delayedPrivatizationInfo.privatizers.begin(),
      delayedPrivatizationInfo.privatizers.end());

  auto parallelOp = genOpWithBody<mlir::omp::ParallelOp>(
      genInfo, clauseOps.ifVar, /*num_threads_var=*/nullptr,
      clauseOps.allocateVars, clauseOps.allocatorVars, clauseOps.reductionVars,
      clauseOps.reductionDeclSymbols.empty()
          ? nullptr
          : firOpBuilder.getArrayAttr(clauseOps.reductionDeclSymbols),
      clauseOps.procBindKindAttr, delayedPrivatizationInfo.originalAddresses,
      delayedPrivatizationInfo.privatizers.empty()
          ? nullptr
          : firOpBuilder.getArrayAttr(privatizers),
      isComposite ? firOpBuilder.getUnitAttr() : nullptr);

  if (clauseOps.numThreadsVar) {
    if (evalNumThreadsOutsideTarget)
      targetOp.getNumThreadsMutable().assign(clauseOps.numThreadsVar);
    else
      parallelOp.getNumThreadsVarMutable().assign(clauseOps.numThreadsVar);
  }

  return parallelOp;
}

static mlir::omp::SectionOp
genSectionOp(Fortran::lower::AbstractConverter &converter,
             Fortran::semantics::SemanticsContext &semaCtx,
             Fortran::lower::pft::Evaluation &eval, bool genNested,
             mlir::Location currentLocation,
             const Fortran::parser::OmpClauseList &clauseList) {
  // Currently only private/firstprivate clause is handled, and
  // all privatization is done within `omp.section` operations.
  return genOpWithBody<mlir::omp::SectionOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setClauses(&clauseList));
}

static mlir::omp::SectionsOp
genSectionsOp(Fortran::lower::AbstractConverter &converter,
              Fortran::semantics::SemanticsContext &semaCtx,
              Fortran::lower::pft::Evaluation &eval,
              mlir::Location currentLocation,
              const SectionsOpClauseOps &clauseOps) {
  return genOpWithBody<mlir::omp::SectionsOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(false),
      /*reduction_vars=*/mlir::ValueRange(), /*reductions=*/nullptr,
      clauseOps.allocateVars, clauseOps.allocatorVars, clauseOps.nowaitAttr);
}

static mlir::omp::SimdLoopOp
genSimdLoopOp(Fortran::lower::AbstractConverter &converter,
              Fortran::semantics::SemanticsContext &semaCtx,
              Fortran::lower::pft::Evaluation &eval, bool isComposite,
              mlir::Location currentLocation,
              const Fortran::parser::OmpClauseList &clauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  SimdLoopOpClauseOps clauseOps;
  genSimdLoopClauses(converter, semaCtx, stmtCtx, clauseList, currentLocation,
                     clauseOps);

  auto *nestedEval =
      getCollapsedLoopEval(eval, Fortran::lower::getCollapseValue(clauseList));

  // TODO Create callback to add reduction vars as entry block arguments.

  // TODO Store clauseOps.reductionVars, clauseOps.reductionDeclSymbols in op.
  return genOpWithBody<mlir::omp::SimdLoopOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, *nestedEval)
          .setGenNested(false),
      /*result_types=*/mlir::TypeRange(), clauseOps.alignedVars,
      clauseOps.alignmentAttrs.empty()
          ? nullptr
          : firOpBuilder.getArrayAttr(clauseOps.alignmentAttrs),
      clauseOps.ifVar, clauseOps.nontemporalVars, clauseOps.orderAttr,
      clauseOps.simdlenAttr, clauseOps.safelenAttr,
      isComposite ? firOpBuilder.getUnitAttr() : nullptr);
}

static mlir::omp::SingleOp
genSingleOp(Fortran::lower::AbstractConverter &converter,
            Fortran::semantics::SemanticsContext &semaCtx,
            Fortran::lower::pft::Evaluation &eval, bool genNested,
            mlir::Location currentLocation,
            const Fortran::parser::OmpClauseList &beginClauseList,
            const Fortran::parser::OmpClauseList &endClauseList) {
  SingleOpClauseOps clauseOps;
  genSingleClauses(converter, semaCtx, beginClauseList, endClauseList,
                   currentLocation, clauseOps);

  return genOpWithBody<mlir::omp::SingleOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setClauses(&beginClauseList),
      clauseOps.allocateVars, clauseOps.allocatorVars,
      clauseOps.copyprivateVars,
      clauseOps.copyprivateFuncs.empty()
          ? nullptr
          : converter.getFirOpBuilder().getArrayAttr(
                clauseOps.copyprivateFuncs),
      clauseOps.nowaitAttr);
}

static mlir::omp::TargetOp
genTargetOp(Fortran::lower::AbstractConverter &converter,
            Fortran::semantics::SemanticsContext &semaCtx,
            Fortran::lower::pft::Evaluation &eval, bool genNested,
            mlir::Location currentLocation,
            const Fortran::parser::OmpClauseList &clauseList,
            bool outerCombined = false) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;

  bool processHostOnlyClauses =
      !llvm::cast<mlir::omp::OffloadModuleInterface>(*converter.getModuleOp())
           .getIsTargetDevice();

  TargetOpClauseOps clauseOps;
  clauseOps.mapSymbols.emplace();
  clauseOps.mapSymLocs.emplace();
  clauseOps.mapSymTypes.emplace();
  genTargetClauses(converter, semaCtx, stmtCtx, clauseList, currentLocation,
                   processHostOnlyClauses, /*processReduction=*/outerCombined,
                   clauseOps);

  // 5.8.1 Implicit Data-Mapping Attribute Rules
  // The following code follows the implicit data-mapping rules to map all the
  // symbols used inside the region that have not been explicitly mapped using
  // the map clause.
  auto captureImplicitMap = [&](const Fortran::semantics::Symbol &sym) {
    if (llvm::find(*clauseOps.mapSymbols, &sym) ==
        clauseOps.mapSymbols->end()) {
      mlir::Value baseOp = converter.getSymbolAddress(sym);
      if (!baseOp)
        if (const auto *details = sym.template detailsIf<
                                  Fortran::semantics::HostAssocDetails>()) {
          baseOp = converter.getSymbolAddress(details->symbol());
          converter.copySymbolBinding(details->symbol(), sym);
        }

      if (baseOp) {
        llvm::SmallVector<mlir::Value> bounds;
        std::stringstream name;
        fir::ExtendedValue dataExv = converter.getSymbolExtendedValue(sym);
        name << sym.name().ToString();

        Fortran::lower::AddrAndBoundsInfo info = getDataOperandBaseAddr(
            converter, firOpBuilder, sym, converter.getCurrentLocation());
        if (fir::unwrapRefType(info.addr.getType()).isa<fir::BaseBoxType>())
          bounds =
              Fortran::lower::genBoundsOpsFromBox<mlir::omp::DataBoundsOp,
                                                  mlir::omp::DataBoundsType>(
                  firOpBuilder, converter.getCurrentLocation(), converter,
                  dataExv, info);
        if (fir::unwrapRefType(info.addr.getType()).isa<fir::SequenceType>()) {
          bool dataExvIsAssumedSize =
              Fortran::semantics::IsAssumedSizeArray(sym.GetUltimate());
          bounds = Fortran::lower::genBaseBoundsOps<mlir::omp::DataBoundsOp,
                                                    mlir::omp::DataBoundsType>(
              firOpBuilder, converter.getCurrentLocation(), converter, dataExv,
              dataExvIsAssumedSize);
        }

        llvm::omp::OpenMPOffloadMappingFlags mapFlag =
            llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_IMPLICIT;
        mlir::omp::VariableCaptureKind captureKind =
            mlir::omp::VariableCaptureKind::ByRef;

        mlir::Type eleType = baseOp.getType();
        if (auto refType = baseOp.getType().dyn_cast<fir::ReferenceType>())
          eleType = refType.getElementType();

        // If a variable is specified in declare target link and if device
        // type is not specified as `nohost`, it needs to be mapped tofrom
        mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
        mlir::Operation *op = mod.lookupSymbol(converter.mangleName(sym));
        auto declareTargetOp =
            llvm::dyn_cast_if_present<mlir::omp::DeclareTargetInterface>(op);
        if (declareTargetOp && declareTargetOp.isDeclareTarget()) {
          if (declareTargetOp.getDeclareTargetCaptureClause() ==
                  mlir::omp::DeclareTargetCaptureClause::link &&
              declareTargetOp.getDeclareTargetDeviceType() !=
                  mlir::omp::DeclareTargetDeviceType::nohost) {
            mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
            mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
          }
        } else if (llvm::find(clauseOps.targetReductionSymbols, &sym) !=
                   clauseOps.targetReductionSymbols.end()) {
          // Do a tofrom map for reduction variables.
          mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
          mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
        } else if (fir::isa_trivial(eleType) || fir::isa_char(eleType)) {
          captureKind = mlir::omp::VariableCaptureKind::ByCopy;
        } else if (!fir::isa_builtin_cptr_type(eleType)) {
          mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO;
          mapFlag |= llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;
        }

        mlir::Value mapOp = createMapInfoOp(
            firOpBuilder, baseOp.getLoc(), baseOp, mlir::Value{}, name.str(),
            bounds, {},
            static_cast<
                std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                mapFlag),
            captureKind, baseOp.getType());

        clauseOps.mapVars.push_back(mapOp);
        clauseOps.mapSymTypes->push_back(baseOp.getType());
        clauseOps.mapSymLocs->push_back(baseOp.getLoc());
        clauseOps.mapSymbols->push_back(&sym);
      }
    }
  };
  Fortran::lower::pft::visitAllSymbols(eval, captureImplicitMap);

  auto targetOp = firOpBuilder.create<mlir::omp::TargetOp>(
      currentLocation, clauseOps.ifVar, clauseOps.deviceVar,
      clauseOps.threadLimitVar,
      /*trip_count=*/nullptr,
      clauseOps.dependTypeAttrs.empty()
          ? nullptr
          : firOpBuilder.getArrayAttr(clauseOps.dependTypeAttrs),
      clauseOps.dependVars, clauseOps.nowaitAttr, clauseOps.mapVars,
      /*num_teams_lower=*/nullptr, /*num_teams_upper=*/nullptr,
      /*teams_thread_limit=*/nullptr, /*num_threads=*/nullptr);

  genBodyOfTargetOp(converter, semaCtx, eval, genNested, targetOp,
                    *clauseOps.mapSymTypes, *clauseOps.mapSymLocs,
                    *clauseOps.mapSymbols, currentLocation);

  return targetOp;
}

static mlir::omp::TaskGroupOp
genTaskGroupOp(Fortran::lower::AbstractConverter &converter,
               Fortran::semantics::SemanticsContext &semaCtx,
               Fortran::lower::pft::Evaluation &eval, bool genNested,
               mlir::Location currentLocation,
               const Fortran::parser::OmpClauseList &clauseList) {
  TaskGroupOpClauseOps clauseOps;
  genTaskGroupClauses(converter, semaCtx, clauseList, currentLocation,
                      clauseOps);

  // TODO Possibly create callback to add task reduction vars as entry block
  // arguments.

  return genOpWithBody<mlir::omp::TaskGroupOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setClauses(&clauseList),
      clauseOps.taskReductionVars,
      clauseOps.taskReductionDeclSymbols.empty()
          ? nullptr
          : converter.getFirOpBuilder().getArrayAttr(
                clauseOps.taskReductionDeclSymbols),
      clauseOps.allocateVars, clauseOps.allocatorVars);
}

static mlir::omp::TaskLoopOp
genTaskLoopOp(Fortran::lower::AbstractConverter &converter,
              Fortran::semantics::SemanticsContext &semaCtx,
              Fortran::lower::pft::Evaluation &eval, bool isComposite,
              mlir::Location currentLocation,
              const Fortran::parser::OmpClauseList &clauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  TaskLoopOpClauseOps clauseOps;
  clauseOps.reductionSymbols.emplace();
  genTaskLoopClauses(converter, semaCtx, stmtCtx, clauseList, currentLocation,
                     clauseOps);

  auto *nestedEval =
      getCollapsedLoopEval(eval, Fortran::lower::getCollapseValue(clauseList));

  auto reductionCallback = [&](mlir::Operation *op) {
    // TODO Possibly add in-reductions to the entry block argument list.
    genReductionVars(op, converter, currentLocation,
                     *clauseOps.reductionSymbols, clauseOps.reductionTypes);
    return *clauseOps.reductionSymbols;
  };

  return genOpWithBody<mlir::omp::TaskLoopOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, *nestedEval)
          .setGenRegionEntryCb(reductionCallback)
          .setReductions(*clauseOps.reductionSymbols, clauseOps.reductionTypes)
          .setGenNested(false),
      clauseOps.ifVar, clauseOps.finalVar, clauseOps.untiedAttr,
      clauseOps.mergeableAttr, clauseOps.inReductionVars,
      clauseOps.inReductionDeclSymbols.empty()
          ? nullptr
          : firOpBuilder.getArrayAttr(clauseOps.inReductionDeclSymbols),
      clauseOps.reductionVars,
      clauseOps.reductionDeclSymbols.empty()
          ? nullptr
          : firOpBuilder.getArrayAttr(clauseOps.reductionDeclSymbols),
      clauseOps.priorityVar, clauseOps.allocateVars, clauseOps.allocatorVars,
      clauseOps.grainsizeVar, clauseOps.numTasksVar, clauseOps.nogroupAttr,
      isComposite ? firOpBuilder.getUnitAttr() : nullptr);
}

static mlir::omp::TaskOp
genTaskOp(Fortran::lower::AbstractConverter &converter,
          Fortran::semantics::SemanticsContext &semaCtx,
          Fortran::lower::pft::Evaluation &eval, bool genNested,
          mlir::Location currentLocation,
          const Fortran::parser::OmpClauseList &clauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  TaskOpClauseOps clauseOps;
  genTaskClauses(converter, semaCtx, stmtCtx, clauseList, currentLocation,
                 clauseOps);

  // TODO Possibly create callback to add in-reductions as entry block
  // arguments.

  return genOpWithBody<mlir::omp::TaskOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setClauses(&clauseList),
      clauseOps.ifVar, clauseOps.finalVar, clauseOps.untiedAttr,
      clauseOps.mergeableAttr, clauseOps.inReductionVars,
      clauseOps.inReductionDeclSymbols.empty()
          ? nullptr
          : firOpBuilder.getArrayAttr(clauseOps.inReductionDeclSymbols),
      clauseOps.priorityVar,
      clauseOps.dependTypeAttrs.empty()
          ? nullptr
          : firOpBuilder.getArrayAttr(clauseOps.dependTypeAttrs),
      clauseOps.dependVars, clauseOps.allocateVars, clauseOps.allocatorVars);
}

static mlir::omp::TaskWaitOp
genTaskWaitOp(Fortran::lower::AbstractConverter &converter,
              Fortran::semantics::SemanticsContext &semaCtx,
              Fortran::lower::pft::Evaluation &eval,
              mlir::Location currentLocation,
              const Fortran::parser::OmpClauseList &clauseList) {
  TaskWaitOpClauseOps clauseOps;
  genTaskWaitClauses(converter, semaCtx, clauseList, currentLocation,
                     clauseOps);
  return converter.getFirOpBuilder().create<mlir::omp::TaskWaitOp>(
      currentLocation);
}

static mlir::omp::TaskYieldOp
genTaskYieldOp(Fortran::lower::AbstractConverter &converter,
               Fortran::semantics::SemanticsContext &semaCtx,
               Fortran::lower::pft::Evaluation &eval,
               mlir::Location currentLocation) {
  return converter.getFirOpBuilder().create<mlir::omp::TaskYieldOp>(
      currentLocation);
}

static mlir::omp::TeamsOp
genTeamsOp(Fortran::lower::AbstractConverter &converter,
           Fortran::semantics::SemanticsContext &semaCtx,
           Fortran::lower::pft::Evaluation &eval, bool genNested,
           mlir::Location currentLocation,
           const Fortran::parser::OmpClauseList &clauseList,
           bool outerCombined = false) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  TeamsOpClauseOps clauseOps;

  auto offloadModOp = llvm::cast<mlir::omp::OffloadModuleInterface>(
      converter.getModuleOp().getOperation());
  mlir::omp::TargetOp targetOp = findParentTargetOp(firOpBuilder);

  bool evalNumTeamsOutsideTarget =
      targetOp && !offloadModOp.getIsTargetDevice();

  genTeamsClauses(converter, semaCtx, stmtCtx, clauseList, currentLocation,
                  evalNumTeamsOutsideTarget, clauseOps);

  // TODO Possibly create callback to add reductions as entry block arguments.

  auto teamsOp = genOpWithBody<mlir::omp::TeamsOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, eval)
          .setGenNested(genNested)
          .setOuterCombined(outerCombined)
          .setClauses(&clauseList),
      /*num_teams_lower=*/nullptr, /*num_teams_upper=*/nullptr, clauseOps.ifVar,
      /*thread_limit=*/nullptr, clauseOps.allocateVars, clauseOps.allocatorVars,
      clauseOps.reductionVars,
      clauseOps.reductionDeclSymbols.empty()
          ? nullptr
          : firOpBuilder.getArrayAttr(clauseOps.reductionDeclSymbols));

  // TODO Populate lower bound once supported by the clause processor
  if (evalNumTeamsOutsideTarget) {
    if (clauseOps.numTeamsUpperVar)
      targetOp.getNumTeamsUpperMutable().assign(clauseOps.numTeamsUpperVar);
    if (clauseOps.threadLimitVar)
      targetOp.getTeamsThreadLimitMutable().assign(clauseOps.threadLimitVar);
  } else {
    if (clauseOps.numTeamsUpperVar)
      teamsOp.getNumTeamsUpperMutable().assign(clauseOps.numTeamsUpperVar);
    if (clauseOps.threadLimitVar)
      teamsOp.getThreadLimitMutable().assign(clauseOps.threadLimitVar);
  }

  return teamsOp;
}

static mlir::omp::WsLoopOp
genWsLoopOp(Fortran::lower::AbstractConverter &converter,
            Fortran::semantics::SemanticsContext &semaCtx,
            Fortran::lower::pft::Evaluation &eval, bool isComposite,
            mlir::Location currentLocation,
            const Fortran::parser::OmpClauseList &beginClauseList,
            const Fortran::parser::OmpClauseList *endClauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  Fortran::lower::StatementContext stmtCtx;
  WsloopOpClauseOps clauseOps;
  clauseOps.reductionSymbols.emplace();
  genWsLoopClauses(converter, semaCtx, stmtCtx, beginClauseList, endClauseList,
                   currentLocation, clauseOps);

  auto *nestedEval = getCollapsedLoopEval(
      eval, Fortran::lower::getCollapseValue(beginClauseList));

  auto reductionCallback = [&](mlir::Operation *op) {
    genReductionVars(op, converter, currentLocation,
                     *clauseOps.reductionSymbols, clauseOps.reductionTypes);
    return *clauseOps.reductionSymbols;
  };

  return genOpWithBody<mlir::omp::WsLoopOp>(
      OpWithBodyGenInfo(converter, semaCtx, currentLocation, *nestedEval)
          .setReductions(*clauseOps.reductionSymbols, clauseOps.reductionTypes)
          .setGenRegionEntryCb(reductionCallback)
          .setGenNested(false),
      clauseOps.linearVars, clauseOps.linearStepVars, clauseOps.reductionVars,
      clauseOps.reductionDeclSymbols.empty()
          ? nullptr
          : firOpBuilder.getArrayAttr(clauseOps.reductionDeclSymbols),
      clauseOps.scheduleValAttr, clauseOps.scheduleChunkVar,
      clauseOps.scheduleModAttr, clauseOps.scheduleSimdAttr,
      clauseOps.nowaitAttr, clauseOps.orderedAttr, clauseOps.orderAttr,
      isComposite ? firOpBuilder.getUnitAttr() : nullptr);
}

//===----------------------------------------------------------------------===//
// Code generation functions for composite constructs
//===----------------------------------------------------------------------===//

static void genCompositeDistributeParallelDo(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OmpClauseList &beginClauseList,
    const Fortran::parser::OmpClauseList *endClauseList,
    mlir::Location currentLocation) {
  Fortran::lower::StatementContext stmtCtx;
  DistributeOpClauseOps distributeClauseOps;
  ParallelOpClauseOps parallelClauseOps;
  WsloopOpClauseOps wsLoopClauseOps;

  genDistributeClauses(converter, semaCtx, beginClauseList, currentLocation,
                       distributeClauseOps);
  // TODO evalNumThreadsOutsideTarget
  genParallelClauses(converter, semaCtx, stmtCtx, beginClauseList,
                     currentLocation, /*processReduction=*/true,
                     /*evalNumThreadsOutsideTarget=*/true, parallelClauseOps);
  genWsLoopClauses(converter, semaCtx, stmtCtx, beginClauseList, endClauseList,
                   currentLocation, wsLoopClauseOps);

  // TODO Pass clauseOps structures to generate wrappers
  // genDistributeOp();
  // genParallelOp();
  // genWsLoopOp();
  TODO(currentLocation, "Composite DISTRIBUTE PARALLEL DO not implemented");
}

static void genCompositeDistributeParallelDoSimd(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OmpClauseList &beginClauseList,
    const Fortran::parser::OmpClauseList *endClauseList,
    mlir::Location currentLocation) {
  Fortran::lower::StatementContext stmtCtx;
  DistributeOpClauseOps distributeClauseOps;
  ParallelOpClauseOps parallelClauseOps;
  WsloopOpClauseOps wsLoopClauseOps;
  SimdLoopOpClauseOps simdClauseOps;

  genDistributeClauses(converter, semaCtx, beginClauseList, currentLocation,
                       distributeClauseOps);
  // TODO evalNumThreadsOutsideTarget
  genParallelClauses(converter, semaCtx, stmtCtx, beginClauseList,
                     currentLocation, /*processReduction=*/true,
                     /*evalNumThreadsOutsideTarget=*/true, parallelClauseOps);
  genWsLoopClauses(converter, semaCtx, stmtCtx, beginClauseList, endClauseList,
                   currentLocation, wsLoopClauseOps);
  genSimdLoopClauses(converter, semaCtx, stmtCtx, beginClauseList,
                     currentLocation, simdClauseOps);

  // TODO Pass clauseOps structures to generate wrappers
  // genDistributeOp();
  // genParallelOp();
  // genWsloopOp();
  // genSimdLoopOp();
  TODO(currentLocation,
       "Composite DISTRIBUTE PARALLEL DO SIMD not implemented");
}

static void genCompositeDistributeSimd(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OmpClauseList &beginClauseList,
    const Fortran::parser::OmpClauseList *endClauseList,
    mlir::Location currentLocation) {
  Fortran::lower::StatementContext stmtCtx;
  DistributeOpClauseOps distributeClauseOps;
  SimdLoopOpClauseOps simdClauseOps;

  genDistributeClauses(converter, semaCtx, beginClauseList, currentLocation,
                       distributeClauseOps);
  genSimdLoopClauses(converter, semaCtx, stmtCtx, beginClauseList,
                     currentLocation, simdClauseOps);

  // TODO Pass clauseOps structures to generate wrappers
  // genDistributeOp();
  // genSimdLoopOp();
  TODO(currentLocation, "Composite DISTRIBUTE SIMD not implemented");
}

static void
genCompositeDoSimd(Fortran::lower::AbstractConverter &converter,
                   Fortran::semantics::SemanticsContext &semaCtx,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OmpClauseList &beginClauseList,
                   const Fortran::parser::OmpClauseList *endClauseList,
                   mlir::Location currentLocation) {
  Fortran::lower::StatementContext stmtCtx;
  WsloopOpClauseOps wsLoopClauseOps;
  SimdLoopOpClauseOps simdClauseOps;

  genWsLoopClauses(converter, semaCtx, stmtCtx, beginClauseList, endClauseList,
                   currentLocation, wsLoopClauseOps);
  genSimdLoopClauses(converter, semaCtx, stmtCtx, beginClauseList,
                     currentLocation, simdClauseOps);

  // TODO Pass clauseOps structures to generate wrappers
  // genWsloopOp();
  // genSimdLoopOp();
  TODO(currentLocation, "Composite DO SIMD not implemented");
}

static void
genCompositeTaskLoopSimd(Fortran::lower::AbstractConverter &converter,
                         Fortran::semantics::SemanticsContext &semaCtx,
                         Fortran::lower::pft::Evaluation &eval,
                         const Fortran::parser::OmpClauseList &beginClauseList,
                         const Fortran::parser::OmpClauseList *endClauseList,
                         mlir::Location currentLocation) {
  Fortran::lower::StatementContext stmtCtx;
  TaskLoopOpClauseOps taskLoopClauseOps;
  SimdLoopOpClauseOps simdClauseOps;

  genTaskLoopClauses(converter, semaCtx, stmtCtx, beginClauseList,
                     currentLocation, taskLoopClauseOps);
  genSimdLoopClauses(converter, semaCtx, stmtCtx, beginClauseList,
                     currentLocation, simdClauseOps);

  // TODO Pass clauseOps structures to generate wrappers
  // genTaskloopOp();
  // genSimdLoopOp();
  TODO(currentLocation, "Composite TASKLOOP SIMD not implemented");
}

//===----------------------------------------------------------------------===//
// genOMP() Code generation functions
//===----------------------------------------------------------------------===//

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::SymMap &symTable,
                   Fortran::semantics::SemanticsContext &semaCtx,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPSimpleStandaloneConstruct
                       &simpleStandaloneConstruct) {
  const auto &directive =
      std::get<Fortran::parser::OmpSimpleStandaloneDirective>(
          simpleStandaloneConstruct.t);
  const auto &clauseList =
      std::get<Fortran::parser::OmpClauseList>(simpleStandaloneConstruct.t);
  mlir::Location currentLocation = converter.genLocation(directive.source);

  switch (directive.v) {
  default:
    break;
  case llvm::omp::Directive::OMPD_barrier:
    genBarrierOp(converter, semaCtx, eval, currentLocation);
    break;
  case llvm::omp::Directive::OMPD_taskwait:
    genTaskWaitOp(converter, semaCtx, eval, currentLocation, clauseList);
    break;
  case llvm::omp::Directive::OMPD_taskyield:
    genTaskYieldOp(converter, semaCtx, eval, currentLocation);
    break;
  case llvm::omp::Directive::OMPD_target_data:
    genDataOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
              clauseList);
    break;
  case llvm::omp::Directive::OMPD_target_enter_data:
    genEnterExitUpdateDataOp<mlir::omp::EnterDataOp>(
        converter, semaCtx, currentLocation, clauseList);
    break;
  case llvm::omp::Directive::OMPD_target_exit_data:
    genEnterExitUpdateDataOp<mlir::omp::ExitDataOp>(
        converter, semaCtx, currentLocation, clauseList);
    break;
  case llvm::omp::Directive::OMPD_target_update:
    genEnterExitUpdateDataOp<mlir::omp::UpdateDataOp>(
        converter, semaCtx, currentLocation, clauseList);
    break;
  case llvm::omp::Directive::OMPD_ordered:
    TODO(currentLocation, "OMPD_ordered");
  }
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPFlushConstruct &flushConstruct) {
  const auto &verbatim = std::get<Fortran::parser::Verbatim>(flushConstruct.t);
  const auto &objectList =
      std::get<std::optional<Fortran::parser::OmpObjectList>>(flushConstruct.t);
  const auto &clauseList =
      std::get<std::optional<std::list<Fortran::parser::OmpMemoryOrderClause>>>(
          flushConstruct.t);
  mlir::Location currentLocation = converter.genLocation(verbatim.source);
  genFlushOp(converter, semaCtx, eval, currentLocation, objectList, clauseList);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPStandaloneConstruct &standaloneConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPSimpleStandaloneConstruct
                  &simpleStandaloneConstruct) {
            genOMP(converter, symTable, semaCtx, eval,
                   simpleStandaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPFlushConstruct &flushConstruct) {
            genOMP(converter, symTable, semaCtx, eval, flushConstruct);
          },
          [&](const Fortran::parser::OpenMPCancelConstruct &cancelConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
          },
          [&](const Fortran::parser::OpenMPCancellationPointConstruct
                  &cancellationPointConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPCancelConstruct");
          },
      },
      standaloneConstruct.u);
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::SymMap &symTable,
                   Fortran::semantics::SemanticsContext &semaCtx,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {
  const auto &beginLoopDirective =
      std::get<Fortran::parser::OmpBeginLoopDirective>(loopConstruct.t);
  const auto &beginClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginLoopDirective.t);
  mlir::Location currentLocation =
      converter.genLocation(beginLoopDirective.source);
  llvm::omp::Directive origDirective =
      std::get<Fortran::parser::OmpLoopDirective>(beginLoopDirective.t).v;

  assert(llvm::omp::loopConstructSet.test(origDirective) &&
         "Expected loop construct");

  const auto *endClauseList = [&]() {
    using RetTy = const Fortran::parser::OmpClauseList *;
    if (auto &endLoopDirective =
            std::get<std::optional<Fortran::parser::OmpEndLoopDirective>>(
                loopConstruct.t)) {
      return RetTy(
          &std::get<Fortran::parser::OmpClauseList>((*endLoopDirective).t));
    }
    return RetTy();
  }();

  /// Utility to remove the first leaf construct from a combined loop construct.
  /// Composite constructs are not handled, as they cannot be split in that way.
  auto peelCombinedLoopDirective =
      [](llvm::omp::Directive dir) -> llvm::omp::Directive {
    using D = llvm::omp::Directive;
    switch (dir) {
    case D::OMPD_masked_taskloop:
    case D::OMPD_master_taskloop:
      return D::OMPD_taskloop;
    case D::OMPD_masked_taskloop_simd:
    case D::OMPD_master_taskloop_simd:
      return D::OMPD_taskloop_simd;
    case D::OMPD_parallel_do:
      return D::OMPD_do;
    case D::OMPD_parallel_do_simd:
      return D::OMPD_do_simd;
    case D::OMPD_parallel_masked_taskloop:
      return D::OMPD_masked_taskloop;
    case D::OMPD_parallel_master_taskloop:
      return D::OMPD_master_taskloop;
    case D::OMPD_parallel_masked_taskloop_simd:
      return D::OMPD_masked_taskloop_simd;
    case D::OMPD_parallel_master_taskloop_simd:
      return D::OMPD_master_taskloop_simd;
    case D::OMPD_target_parallel_do:
      return D::OMPD_parallel_do;
    case D::OMPD_target_parallel_do_simd:
      return D::OMPD_parallel_do_simd;
    case D::OMPD_target_simd:
      return D::OMPD_simd;
    case D::OMPD_target_teams_distribute:
      return D::OMPD_teams_distribute;
    case D::OMPD_target_teams_distribute_parallel_do:
      return D::OMPD_teams_distribute_parallel_do;
    case D::OMPD_target_teams_distribute_parallel_do_simd:
      return D::OMPD_teams_distribute_parallel_do_simd;
    case D::OMPD_target_teams_distribute_simd:
      return D::OMPD_teams_distribute_simd;
    case D::OMPD_teams_distribute:
      return D::OMPD_distribute;
    case D::OMPD_teams_distribute_parallel_do:
      return D::OMPD_distribute_parallel_do;
    case D::OMPD_teams_distribute_parallel_do_simd:
      return D::OMPD_distribute_parallel_do_simd;
    case D::OMPD_teams_distribute_simd:
      return D::OMPD_distribute_simd;
    case D::OMPD_parallel_loop:
    case D::OMPD_teams_loop:
      return D::OMPD_loop;
    case D::OMPD_target_parallel_loop:
      return D::OMPD_parallel_loop;
    case D::OMPD_target_teams_loop:
      return D::OMPD_teams_loop;
    default:
      llvm_unreachable("Unexpected non-combined loop construct");
    }
  };

  // Privatization and loop nest clause processing must be done before producing
  // any wrappers and after combined constructs, so that any operations created
  // are outside of the wrapper nest.
  DataSharingProcessor dsp(converter, beginClauseList, eval);
  LoopNestOpClauseOps clauseOps;
  auto processLoopNestClauses = [&]() {
    dsp.processStep1();
    genLoopNestClauses(converter, semaCtx, eval, beginClauseList,
                       currentLocation, clauseOps);
  };

  llvm::omp::Directive ompDirective = origDirective;
  if (llvm::omp::topTargetSet.test(ompDirective)) {
    // TODO Combined constructs: Call gen<X>Clauses and pass them in.
    genTargetOp(converter, semaCtx, eval, /*genNested=*/false, currentLocation,
                beginClauseList, /*outerCombined=*/true);
    ompDirective = peelCombinedLoopDirective(ompDirective);
  }

  if (llvm::omp::topTeamsSet.test(ompDirective)) {
    genTeamsOp(converter, semaCtx, eval, /*genNested=*/false, currentLocation,
               beginClauseList, /*outerCombined=*/true);
    ompDirective = peelCombinedLoopDirective(ompDirective);
  }

  if (llvm::omp::topParallelSet.test(ompDirective)) {
    genParallelOp(converter, symTable, semaCtx, eval, /*genNested=*/false,
                  /*isComposite=*/false, currentLocation, beginClauseList,
                  /*outerCombined=*/true);
    ompDirective = peelCombinedLoopDirective(ompDirective);
    processLoopNestClauses();
  } else {
    processLoopNestClauses();

    if (llvm::omp::topDistributeSet.test(ompDirective)) {
      switch (ompDirective) {
      case llvm::omp::Directive::OMPD_distribute:
        genDistributeOp(converter, semaCtx, eval, /*isComposite=*/false,
                        currentLocation, beginClauseList,
                        /*outerCombined=*/true);
        break;
      case llvm::omp::Directive::OMPD_distribute_parallel_do:
        genCompositeDistributeParallelDo(converter, semaCtx, eval,
                                         beginClauseList, endClauseList,
                                         currentLocation);
        break;
      case llvm::omp::Directive::OMPD_distribute_parallel_do_simd:
        genCompositeDistributeParallelDoSimd(converter, semaCtx, eval,
                                             beginClauseList, endClauseList,
                                             currentLocation);
        break;
      case llvm::omp::Directive::OMPD_distribute_simd:
        genCompositeDistributeSimd(converter, semaCtx, eval, beginClauseList,
                                   endClauseList, currentLocation);
        break;
      default:
        llvm_unreachable("Unexpected DISTRIBUTE construct");
      }
    } else if (llvm::omp::topTaskloopSet.test(ompDirective)) {
      switch (ompDirective) {
      case llvm::omp::Directive::OMPD_taskloop_simd:
        genCompositeTaskLoopSimd(converter, semaCtx, eval, beginClauseList,
                                 endClauseList, currentLocation);
        break;
      case llvm::omp::Directive::OMPD_taskloop:
        genTaskLoopOp(converter, semaCtx, eval, /*isComposite=*/false,
                      currentLocation, beginClauseList);
        break;
      default:
        llvm_unreachable("Unexpected TASKLOOP construct");
      }
    } else if (ompDirective == llvm::omp::Directive::OMPD_simd) {
      genSimdLoopOp(converter, semaCtx, eval, /*isComposite=*/false,
                    currentLocation, beginClauseList);
    } else if (!llvm::omp::topDoSet.test(ompDirective)) {
      TODO(currentLocation,
           "Unhandled loop directive (" +
               llvm::omp::getOpenMPDirectiveName(origDirective) + ")");
    }
  }

  if (llvm::omp::topDoSet.test(ompDirective)) {
    switch (ompDirective) {
    case llvm::omp::Directive::OMPD_do_simd:
      genCompositeDoSimd(converter, semaCtx, eval, beginClauseList,
                         endClauseList, currentLocation);
      break;
    case llvm::omp::Directive::OMPD_do:
      genWsLoopOp(converter, semaCtx, eval, /*isComposite=*/false,
                  currentLocation, beginClauseList, endClauseList);
      break;
    default:
      llvm_unreachable("Unexpected DO construct");
    }
  } else if (llvm::omp::allParallelSet.test(origDirective)) {
    TODO(currentLocation, "Unhandled loop directive (" +
                              llvm::omp::getOpenMPDirectiveName(origDirective) +
                              ")");
  }

  // Create inner loop nest and body.
  mlir::omp::LoopNestOp loopNestOp =
      genLoopNestOp(converter, semaCtx, eval, currentLocation, beginClauseList,
                    clauseOps, dsp);

  if (ompDirective == llvm::omp::Directive::OMPD_simd)
    genOpenMPReduction(converter, semaCtx, beginClauseList);

  // Create trip_count outside of omp.target if this is host compilation and the
  // loop is inside of a target region.
  auto offloadMod = llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(
      converter.getModuleOp().getOperation());
  auto targetOp = loopNestOp->getParentOfType<mlir::omp::TargetOp>();

  if (offloadMod && targetOp && !offloadMod.getIsTargetDevice() &&
      targetOp.isTargetSPMDLoop()) {
    // Lower loop bounds and step, and process collapsing again, putting lowered
    // values outside of omp.target this time. This enables calculating and
    // accessing the trip count in the host, which is needed when lowering to
    // LLVM IR via the OMPIRBuilder.
    HostClausesInsertionGuard guard(converter.getFirOpBuilder());
    CollapseClauseOps collapseOps;
    ClauseProcessor(converter, semaCtx, beginClauseList)
        .processCollapse(currentLocation, eval, collapseOps);
    targetOp.getTripCountMutable().assign(
        calculateTripCount(converter, currentLocation, collapseOps.loopLBVar,
                           collapseOps.loopUBVar, collapseOps.loopStepVar));
  }
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
  const auto &beginBlockDirective =
      std::get<Fortran::parser::OmpBeginBlockDirective>(blockConstruct.t);
  const auto &endBlockDirective =
      std::get<Fortran::parser::OmpEndBlockDirective>(blockConstruct.t);
  const auto &directive =
      std::get<Fortran::parser::OmpBlockDirective>(beginBlockDirective.t);
  const auto &beginClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginBlockDirective.t);
  const auto &endClauseList =
      std::get<Fortran::parser::OmpClauseList>(endBlockDirective.t);

  assert(llvm::omp::blockConstructSet.test(directive.v) &&
         "Expected block construct");

  for (const Fortran::parser::OmpClause &clause : beginClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (!std::get_if<Fortran::parser::OmpClause::If>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::NumThreads>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::ProcBind>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Allocate>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Default>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Final>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Priority>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Reduction>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Depend>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Private>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Firstprivate>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Copyin>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Shared>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Threads>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Map>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::UseDevicePtr>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::UseDeviceAddr>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::ThreadLimit>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::NumTeams>(&clause.u)) {
      TODO(clauseLocation, "OpenMP Block construct clause");
    }
  }

  for (const auto &clause : endClauseList.v) {
    mlir::Location clauseLocation = converter.genLocation(clause.source);
    if (!std::get_if<Fortran::parser::OmpClause::Nowait>(&clause.u) &&
        !std::get_if<Fortran::parser::OmpClause::Copyprivate>(&clause.u))
      TODO(clauseLocation, "OpenMP Block construct clause");
  }

  bool singleDirective = true;
  mlir::Location currentLocation = converter.genLocation(directive.source);
  switch (directive.v) {
  case llvm::omp::Directive::OMPD_master:
    genMasterOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation);
    break;
  case llvm::omp::Directive::OMPD_ordered:
    genOrderedRegionOp(converter, semaCtx, eval, /*genNested=*/true,
                       currentLocation, beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_parallel:
    genParallelOp(converter, symTable, semaCtx, eval, /*genNested=*/true,
                  /*isComposite=*/false, currentLocation, beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_single:
    genSingleOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
                beginClauseList, endClauseList);
    break;
  case llvm::omp::Directive::OMPD_target:
    genTargetOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
                beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_target_data:
    genDataOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
              beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_task:
    genTaskOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
              beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_taskgroup:
    genTaskGroupOp(converter, semaCtx, eval, /*genNested=*/true,
                   currentLocation, beginClauseList);
    break;
  case llvm::omp::Directive::OMPD_teams:
    genTeamsOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
               beginClauseList, /*outerCombined=*/false);
    break;
  case llvm::omp::Directive::OMPD_workshare:
    // FIXME: Workshare is not a commonly used OpenMP construct, an
    // implementation for this feature will come later. For the codes
    // that use this construct, add a single construct for now.
    genSingleOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
                beginClauseList, endClauseList);
    break;
  default:
    singleDirective = false;
    break;
  }

  if (singleDirective)
    return;

  // Codegen for combined directives
  bool combinedDirective = false;
  if (llvm::omp::allTargetSet.test(directive.v)) {
    genTargetOp(converter, semaCtx, eval, /*genNested=*/false, currentLocation,
                beginClauseList, /*outerCombined=*/true);
    combinedDirective = true;
  }
  if (llvm::omp::allTeamsSet.test(directive.v)) {
    genTeamsOp(converter, semaCtx, eval, /*genNested=*/false, currentLocation,
               beginClauseList);
    combinedDirective = true;
  }
  if (llvm::omp::allParallelSet.test(directive.v)) {
    bool outerCombined =
        directive.v != llvm::omp::Directive::OMPD_target_parallel;
    genParallelOp(converter, symTable, semaCtx, eval, /*genNested=*/false,
                  /*isComposite=*/false, currentLocation, beginClauseList,
                  outerCombined);
    combinedDirective = true;
  }
  if (llvm::omp::workShareSet.test(directive.v)) {
    genSingleOp(converter, semaCtx, eval, /*genNested=*/false, currentLocation,
                beginClauseList, endClauseList);
    combinedDirective = true;
  }
  if (!combinedDirective)
    TODO(currentLocation, "Unhandled block directive (" +
                              llvm::omp::getOpenMPDirectiveName(directive.v) +
                              ")");

  genNestedEvaluations(converter, eval);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPCriticalConstruct &criticalConstruct) {
  const auto &cd =
      std::get<Fortran::parser::OmpCriticalDirective>(criticalConstruct.t);
  const auto &clauseList = std::get<Fortran::parser::OmpClauseList>(cd.t);
  const auto &name = std::get<std::optional<Fortran::parser::Name>>(cd.t);
  mlir::Location currentLocation = converter.getCurrentLocation();
  genCriticalOp(converter, semaCtx, eval, /*genNested=*/true, currentLocation,
                clauseList, name);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPSectionsConstruct &sectionsConstruct) {
  const auto &beginSectionsDirective =
      std::get<Fortran::parser::OmpBeginSectionsDirective>(sectionsConstruct.t);
  const auto &beginClauseList =
      std::get<Fortran::parser::OmpClauseList>(beginSectionsDirective.t);
  llvm::omp::Directive dir =
      std::get<Fortran::parser::OmpSectionsDirective>(beginSectionsDirective.t)
          .v;
  const auto &sectionBlocks =
      std::get<Fortran::parser::OmpSectionBlocks>(sectionsConstruct.t);

  // Process clauses before optional omp.parallel, so that new variables are
  // allocated outside of the parallel region.
  mlir::Location currentLocation = converter.getCurrentLocation();
  SectionsOpClauseOps clauseOps;
  genSectionsClauses(converter, semaCtx, beginClauseList, currentLocation,
                     /*clausesFromBeginSections=*/true, clauseOps);

  // Parallel wrapper of PARALLEL SECTIONS construct.
  if (dir == llvm::omp::Directive::OMPD_parallel_sections) {
    genParallelOp(converter, symTable, semaCtx, eval,
                  /*genNested=*/false, /*isComposite=*/false, currentLocation,
                  beginClauseList, /*outerCombined=*/true);
  } else {
    const auto &endSectionsDirective =
        std::get<Fortran::parser::OmpEndSectionsDirective>(sectionsConstruct.t);
    const auto &endClauseList =
        std::get<Fortran::parser::OmpClauseList>(endSectionsDirective.t);
    genSectionsClauses(converter, semaCtx, endClauseList, currentLocation,
                       /*clausesFromBeginSections=*/false, clauseOps);
  }

  // SECTIONS construct.
  genSectionsOp(converter, semaCtx, eval, currentLocation, clauseOps);

  // Generate nested SECTION operations recursively.
  auto &firOpBuilder = converter.getFirOpBuilder();
  auto ip = firOpBuilder.saveInsertionPoint();
  for (const auto &[nblock, neval] :
       llvm::zip(sectionBlocks.v, eval.getNestedEvaluations())) {
    symTable.pushScope();
    genSectionOp(converter, semaCtx, neval, /*genNested=*/true, currentLocation,
                 beginClauseList);
    symTable.popScope();
    firOpBuilder.restoreInsertionPoint(ip);
  }
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPAtomicConstruct &atomicConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OmpAtomicRead &atomicRead) {
            mlir::Location loc = converter.genLocation(atomicRead.source);
            Fortran::lower::genOmpAccAtomicRead<
                Fortran::parser::OmpAtomicRead,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicRead,
                                                      loc);
          },
          [&](const Fortran::parser::OmpAtomicWrite &atomicWrite) {
            mlir::Location loc = converter.genLocation(atomicWrite.source);
            Fortran::lower::genOmpAccAtomicWrite<
                Fortran::parser::OmpAtomicWrite,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicWrite,
                                                      loc);
          },
          [&](const Fortran::parser::OmpAtomic &atomicConstruct) {
            mlir::Location loc = converter.genLocation(atomicConstruct.source);
            Fortran::lower::genOmpAtomic<Fortran::parser::OmpAtomic,
                                         Fortran::parser::OmpAtomicClauseList>(
                converter, atomicConstruct, loc);
          },
          [&](const Fortran::parser::OmpAtomicUpdate &atomicUpdate) {
            mlir::Location loc = converter.genLocation(atomicUpdate.source);
            Fortran::lower::genOmpAccAtomicUpdate<
                Fortran::parser::OmpAtomicUpdate,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicUpdate,
                                                      loc);
          },
          [&](const Fortran::parser::OmpAtomicCapture &atomicCapture) {
            mlir::Location loc = converter.genLocation(atomicCapture.source);
            Fortran::lower::genOmpAccAtomicCapture<
                Fortran::parser::OmpAtomicCapture,
                Fortran::parser::OmpAtomicClauseList>(converter, atomicCapture,
                                                      loc);
          },
      },
      atomicConstruct.u);
}

static void
markDeclareTarget(mlir::Operation *op,
                  Fortran::lower::AbstractConverter &converter,
                  mlir::omp::DeclareTargetCaptureClause captureClause,
                  mlir::omp::DeclareTargetDeviceType deviceType) {
  // TODO: Add support for program local variables with declare target applied
  auto declareTargetOp = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(op);
  if (!declareTargetOp)
    fir::emitFatalError(
        converter.getCurrentLocation(),
        "Attempt to apply declare target on unsupported operation");

  // The function or global already has a declare target applied to it, very
  // likely through implicit capture (usage in another declare target
  // function/subroutine). It should be marked as any if it has been assigned
  // both host and nohost, else we skip, as there is no change
  if (declareTargetOp.isDeclareTarget()) {
    if (declareTargetOp.getDeclareTargetDeviceType() != deviceType)
      declareTargetOp.setDeclareTarget(mlir::omp::DeclareTargetDeviceType::any,
                                       captureClause);
    return;
  }

  declareTargetOp.setDeclareTarget(deviceType, captureClause);
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::SymMap &symTable,
                   Fortran::semantics::SemanticsContext &semaCtx,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPDeclareTargetConstruct
                       &declareTargetConstruct) {
  mlir::ModuleOp mod = converter.getFirOpBuilder().getModule();
  DeclareTargetOpClauseOps clauseOps;
  mlir::omp::DeclareTargetDeviceType deviceType = getDeclareTargetInfo(
      converter, semaCtx, eval, declareTargetConstruct, clauseOps);

  for (const DeclareTargetCapturePair &symClause : clauseOps.symbolAndClause) {
    mlir::Operation *op = mod.lookupSymbol(converter.mangleName(
        std::get<const Fortran::semantics::Symbol &>(symClause)));

    // Some symbols are deferred until later in the module, these are handled
    // upon finalization of the module for OpenMP inside of Bridge, so we simply
    // skip for now.
    if (!op)
      continue;

    markDeclareTarget(
        op, converter,
        std::get<mlir::omp::DeclareTargetCaptureClause>(symClause), deviceType);
  }
}

static void genOMP(Fortran::lower::AbstractConverter &converter,
                   Fortran::lower::SymMap &symTable,
                   Fortran::semantics::SemanticsContext &semaCtx,
                   Fortran::lower::pft::Evaluation &eval,
                   const Fortran::parser::OpenMPConstruct &ompConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPStandaloneConstruct
                  &standaloneConstruct) {
            genOMP(converter, symTable, semaCtx, eval, standaloneConstruct);
          },
          [&](const Fortran::parser::OpenMPSectionsConstruct
                  &sectionsConstruct) {
            genOMP(converter, symTable, semaCtx, eval, sectionsConstruct);
          },
          [&](const Fortran::parser::OpenMPSectionConstruct &sectionConstruct) {
            // SECTION constructs are handled as a part of SECTIONS.
            llvm_unreachable("Unexpected standalone OMP SECTION");
          },
          [&](const Fortran::parser::OpenMPLoopConstruct &loopConstruct) {
            genOMP(converter, symTable, semaCtx, eval, loopConstruct);
          },
          [&](const Fortran::parser::OpenMPDeclarativeAllocate
                  &execAllocConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
          },
          [&](const Fortran::parser::OpenMPExecutableAllocate
                  &execAllocConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPExecutableAllocate");
          },
          [&](const Fortran::parser::OpenMPAllocatorsConstruct
                  &allocsConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPAllocatorsConstruct");
          },
          [&](const Fortran::parser::OpenMPBlockConstruct &blockConstruct) {
            genOMP(converter, symTable, semaCtx, eval, blockConstruct);
          },
          [&](const Fortran::parser::OpenMPAtomicConstruct &atomicConstruct) {
            genOMP(converter, symTable, semaCtx, eval, atomicConstruct);
          },
          [&](const Fortran::parser::OpenMPCriticalConstruct
                  &criticalConstruct) {
            genOMP(converter, symTable, semaCtx, eval, criticalConstruct);
          },
      },
      ompConstruct.u);
}

static void
genOMP(Fortran::lower::AbstractConverter &converter,
       Fortran::lower::SymMap &symTable,
       Fortran::semantics::SemanticsContext &semaCtx,
       Fortran::lower::pft::Evaluation &eval,
       const Fortran::parser::OpenMPDeclarativeConstruct &ompDeclConstruct) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPDeclarativeAllocate
                  &declarativeAllocate) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclarativeAllocate");
          },
          [&](const Fortran::parser::OpenMPDeclareReductionConstruct
                  &declareReductionConstruct) {
            TODO(converter.getCurrentLocation(),
                 "OpenMPDeclareReductionConstruct");
          },
          [&](const Fortran::parser::OpenMPDeclareSimdConstruct
                  &declareSimdConstruct) {
            TODO(converter.getCurrentLocation(), "OpenMPDeclareSimdConstruct");
          },
          [&](const Fortran::parser::OpenMPDeclareTargetConstruct
                  &declareTargetConstruct) {
            genOMP(converter, symTable, semaCtx, eval, declareTargetConstruct);
          },
          [&](const Fortran::parser::OpenMPRequiresConstruct
                  &requiresConstruct) {
            // Requires directives are gathered and processed in semantics and
            // then combined in the lowering bridge before triggering codegen
            // just once. Hence, there is no need to lower each individual
            // occurrence here.
          },
          [&](const Fortran::parser::OpenMPThreadprivate &threadprivate) {
            // The directive is lowered when instantiating the variable to
            // support the case of threadprivate variable declared in module.
          },
      },
      ompDeclConstruct.u);
}

//===----------------------------------------------------------------------===//
// Public functions
//===----------------------------------------------------------------------===//

mlir::Operation *Fortran::lower::genOpenMPTerminator(fir::FirOpBuilder &builder,
                                                     mlir::Operation *op,
                                                     mlir::Location loc) {
  if (mlir::isa<mlir::omp::LoopNestOp, mlir::omp::ReductionDeclareOp,
                mlir::omp::AtomicUpdateOp>(op))
    return builder.create<mlir::omp::YieldOp>(loc);
  return builder.create<mlir::omp::TerminatorOp>(loc);
}

void Fortran::lower::genOpenMPConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symTable,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPConstruct &omp) {
  symTable.pushScope();
  genOMP(converter, symTable, semaCtx, eval, omp);
  symTable.popScope();
}

void Fortran::lower::genOpenMPDeclarativeConstruct(
    Fortran::lower::AbstractConverter &converter,
    Fortran::lower::SymMap &symTable,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclarativeConstruct &omp) {
  genOMP(converter, symTable, semaCtx, eval, omp);
  genNestedEvaluations(converter, eval);
}

void Fortran::lower::genOpenMPSymbolProperties(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var) {
  assert(var.hasSymbol() && "Expecting Symbol");
  const Fortran::semantics::Symbol &sym = var.getSymbol();

  if (sym.test(Fortran::semantics::Symbol::Flag::OmpThreadprivate))
    Fortran::lower::genThreadprivateOp(converter, var);

  if (sym.test(Fortran::semantics::Symbol::Flag::OmpDeclareTarget))
    Fortran::lower::genDeclareTargetIntGlobal(converter, var);
}

int64_t Fortran::lower::getCollapseValue(
    const Fortran::parser::OmpClauseList &clauseList) {
  for (const Fortran::parser::OmpClause &clause : clauseList.v) {
    if (const auto &collapseClause =
            std::get_if<Fortran::parser::OmpClause::Collapse>(&clause.u)) {
      const auto *expr = Fortran::semantics::GetExpr(collapseClause->v);
      return Fortran::evaluate::ToInt64(*expr).value();
    }
  }
  return 1;
}

void Fortran::lower::genThreadprivateOp(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();
  mlir::Location currentLocation = converter.getCurrentLocation();

  const Fortran::semantics::Symbol &sym = var.getSymbol();
  mlir::Value symThreadprivateValue;
  if (const Fortran::semantics::Symbol *common =
          Fortran::semantics::FindCommonBlockContaining(sym.GetUltimate())) {
    mlir::Value commonValue = converter.getSymbolAddress(*common);
    if (mlir::isa<mlir::omp::ThreadprivateOp>(commonValue.getDefiningOp())) {
      // Generate ThreadprivateOp for a common block instead of its members and
      // only do it once for a common block.
      return;
    }
    // Generate ThreadprivateOp and rebind the common block.
    mlir::Value commonThreadprivateValue =
        firOpBuilder.create<mlir::omp::ThreadprivateOp>(
            currentLocation, commonValue.getType(), commonValue);
    converter.bindSymbol(*common, commonThreadprivateValue);
    // Generate the threadprivate value for the common block member.
    symThreadprivateValue = genCommonBlockMember(converter, currentLocation,
                                                 sym, commonThreadprivateValue);
  } else if (!var.isGlobal()) {
    // Non-global variable which can be in threadprivate directive must be one
    // variable in main program, and it has implicit SAVE attribute. Take it as
    // with SAVE attribute, so to create GlobalOp for it to simplify the
    // translation to LLVM IR.
    fir::GlobalOp global = globalInitialization(converter, firOpBuilder, sym,
                                                var, currentLocation);

    mlir::Value symValue = firOpBuilder.create<fir::AddrOfOp>(
        currentLocation, global.resultType(), global.getSymbol());
    symThreadprivateValue = firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  } else {
    mlir::Value symValue = converter.getSymbolAddress(sym);

    // The symbol may be use-associated multiple times, and nothing needs to be
    // done after the original symbol is mapped to the threadprivatized value
    // for the first time. Use the threadprivatized value directly.
    mlir::Operation *op;
    if (auto declOp = symValue.getDefiningOp<hlfir::DeclareOp>())
      op = declOp.getMemref().getDefiningOp();
    else
      op = symValue.getDefiningOp();
    if (mlir::isa<mlir::omp::ThreadprivateOp>(op))
      return;

    symThreadprivateValue = firOpBuilder.create<mlir::omp::ThreadprivateOp>(
        currentLocation, symValue.getType(), symValue);
  }

  fir::ExtendedValue sexv = converter.getSymbolExtendedValue(sym);
  fir::ExtendedValue symThreadprivateExv =
      getExtendedValue(sexv, symThreadprivateValue);
  converter.bindSymbol(sym, symThreadprivateExv);
}

// This function replicates threadprivate's behaviour of generating
// an internal fir.GlobalOp for non-global variables in the main program
// that have the implicit SAVE attribute, to simplifiy LLVM-IR and MLIR
// generation.
void Fortran::lower::genDeclareTargetIntGlobal(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::pft::Variable &var) {
  if (!var.isGlobal()) {
    // A non-global variable which can be in a declare target directive must
    // be a variable in the main program, and it has the implicit SAVE
    // attribute. We create a GlobalOp for it to simplify the translation to
    // LLVM IR.
    globalInitialization(converter, converter.getFirOpBuilder(),
                         var.getSymbol(), var, converter.getCurrentLocation());
  }
}

// Generate an OpenMP reduction operation.
// TODO: Currently assumes it is either an integer addition/multiplication
// reduction, or a logical and reduction. Generalize this for various reduction
// operation types.
// TODO: Generate the reduction operation during lowering instead of creating
// and removing operations since this is not a robust approach. Also, removing
// ops in the builder (instead of a rewriter) is probably not the best approach.
void Fortran::lower::genOpenMPReduction(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    const Fortran::parser::OmpClauseList &clauseList) {
  fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

  for (const Fortran::parser::OmpClause &clause : clauseList.v) {
    if (const auto &reductionClause =
            std::get_if<Fortran::parser::OmpClause::Reduction>(&clause.u)) {
      const auto &redOperator{std::get<Fortran::parser::OmpReductionOperator>(
          reductionClause->v.t)};
      const auto &objectList{
          std::get<Fortran::parser::OmpObjectList>(reductionClause->v.t)};
      if (const auto *reductionOp =
              std::get_if<Fortran::parser::DefinedOperator>(&redOperator.u)) {
        const auto &intrinsicOp{
            std::get<Fortran::parser::DefinedOperator::IntrinsicOperator>(
                reductionOp->u)};

        switch (intrinsicOp) {
        case Fortran::parser::DefinedOperator::IntrinsicOperator::Add:
        case Fortran::parser::DefinedOperator::IntrinsicOperator::Multiply:
        case Fortran::parser::DefinedOperator::IntrinsicOperator::AND:
        case Fortran::parser::DefinedOperator::IntrinsicOperator::EQV:
        case Fortran::parser::DefinedOperator::IntrinsicOperator::OR:
        case Fortran::parser::DefinedOperator::IntrinsicOperator::NEQV:
          break;
        default:
          continue;
        }
        for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
          if (const auto *name{
                  Fortran::parser::Unwrap<Fortran::parser::Name>(ompObject)}) {
            if (const Fortran::semantics::Symbol * symbol{name->symbol}) {
              mlir::Value reductionVal = converter.getSymbolAddress(*symbol);
              if (auto declOp = reductionVal.getDefiningOp<hlfir::DeclareOp>())
                reductionVal = declOp.getBase();
              mlir::Type reductionType =
                  reductionVal.getType().cast<fir::ReferenceType>().getEleTy();
              if (!reductionType.isa<fir::LogicalType>()) {
                if (!reductionType.isIntOrIndexOrFloat())
                  continue;
              }
              for (mlir::OpOperand &reductionValUse : reductionVal.getUses()) {
                if (auto loadOp = mlir::dyn_cast<fir::LoadOp>(
                        reductionValUse.getOwner())) {
                  mlir::Value loadVal = loadOp.getRes();
                  if (reductionType.isa<fir::LogicalType>()) {
                    mlir::Operation *reductionOp = findReductionChain(loadVal);
                    fir::ConvertOp convertOp =
                        getConvertFromReductionOp(reductionOp, loadVal);
                    updateReduction(reductionOp, firOpBuilder, loadVal,
                                    reductionVal, &convertOp);
                    removeStoreOp(reductionOp, reductionVal);
                  } else if (mlir::Operation *reductionOp =
                                 findReductionChain(loadVal, &reductionVal)) {
                    updateReduction(reductionOp, firOpBuilder, loadVal,
                                    reductionVal);
                  }
                }
              }
            }
          }
        }
      } else if (const auto *reductionIntrinsic =
                     std::get_if<Fortran::parser::ProcedureDesignator>(
                         &redOperator.u)) {
        if (!ReductionProcessor::supportedIntrinsicProcReduction(
                *reductionIntrinsic))
          continue;
        ReductionProcessor::ReductionIdentifier redId =
            ReductionProcessor::getReductionType(*reductionIntrinsic);
        for (const Fortran::parser::OmpObject &ompObject : objectList.v) {
          if (const auto *name{
                  Fortran::parser::Unwrap<Fortran::parser::Name>(ompObject)}) {
            if (const Fortran::semantics::Symbol * symbol{name->symbol}) {
              mlir::Value reductionVal = converter.getSymbolAddress(*symbol);
              if (auto declOp = reductionVal.getDefiningOp<hlfir::DeclareOp>())
                reductionVal = declOp.getBase();
              for (const mlir::OpOperand &reductionValUse :
                   reductionVal.getUses()) {
                if (auto loadOp = mlir::dyn_cast<fir::LoadOp>(
                        reductionValUse.getOwner())) {
                  mlir::Value loadVal = loadOp.getRes();
                  // Max is lowered as a compare -> select.
                  // Match the pattern here.
                  mlir::Operation *reductionOp =
                      findReductionChain(loadVal, &reductionVal);
                  if (reductionOp == nullptr)
                    continue;

                  if (redId == ReductionProcessor::ReductionIdentifier::MAX ||
                      redId == ReductionProcessor::ReductionIdentifier::MIN) {
                    assert(mlir::isa<mlir::arith::SelectOp>(reductionOp) &&
                           "Selection Op not found in reduction intrinsic");
                    mlir::Operation *compareOp =
                        getCompareFromReductionOp(reductionOp, loadVal);
                    updateReduction(compareOp, firOpBuilder, loadVal,
                                    reductionVal);
                  }
                  if (redId == ReductionProcessor::ReductionIdentifier::IOR ||
                      redId == ReductionProcessor::ReductionIdentifier::IEOR ||
                      redId == ReductionProcessor::ReductionIdentifier::IAND) {
                    updateReduction(reductionOp, firOpBuilder, loadVal,
                                    reductionVal);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

mlir::Operation *Fortran::lower::findReductionChain(mlir::Value loadVal,
                                                    mlir::Value *reductionVal) {
  for (mlir::OpOperand &loadOperand : loadVal.getUses()) {
    if (mlir::Operation *reductionOp = loadOperand.getOwner()) {
      if (auto convertOp = mlir::dyn_cast<fir::ConvertOp>(reductionOp)) {
        for (mlir::OpOperand &convertOperand : convertOp.getRes().getUses()) {
          if (mlir::Operation *reductionOp = convertOperand.getOwner())
            return reductionOp;
        }
      }
      for (mlir::OpOperand &reductionOperand : reductionOp->getUses()) {
        if (auto store =
                mlir::dyn_cast<fir::StoreOp>(reductionOperand.getOwner())) {
          if (store.getMemref() == *reductionVal) {
            store.erase();
            return reductionOp;
          }
        }
        if (auto assign =
                mlir::dyn_cast<hlfir::AssignOp>(reductionOperand.getOwner())) {
          if (assign.getLhs() == *reductionVal) {
            assign.erase();
            return reductionOp;
          }
        }
      }
    }
  }
  return nullptr;
}

// for a logical operator 'op' reduction X = X op Y
// This function returns the operation responsible for converting Y from
// fir.logical<4> to i1
fir::ConvertOp
Fortran::lower::getConvertFromReductionOp(mlir::Operation *reductionOp,
                                          mlir::Value loadVal) {
  for (mlir::Value reductionOperand : reductionOp->getOperands()) {
    if (auto convertOp =
            mlir::dyn_cast<fir::ConvertOp>(reductionOperand.getDefiningOp())) {
      if (convertOp.getOperand() == loadVal)
        continue;
      return convertOp;
    }
  }
  return nullptr;
}

void Fortran::lower::updateReduction(mlir::Operation *op,
                                     fir::FirOpBuilder &firOpBuilder,
                                     mlir::Value loadVal,
                                     mlir::Value reductionVal,
                                     fir::ConvertOp *convertOp) {
  mlir::OpBuilder::InsertPoint insertPtDel = firOpBuilder.saveInsertionPoint();
  firOpBuilder.setInsertionPoint(op);

  mlir::Value reductionOp;
  if (convertOp)
    reductionOp = convertOp->getOperand();
  else if (op->getOperand(0) == loadVal)
    reductionOp = op->getOperand(1);
  else
    reductionOp = op->getOperand(0);

  firOpBuilder.create<mlir::omp::ReductionOp>(op->getLoc(), reductionOp,
                                              reductionVal);
  firOpBuilder.restoreInsertionPoint(insertPtDel);
}

void Fortran::lower::removeStoreOp(mlir::Operation *reductionOp,
                                   mlir::Value symVal) {
  for (mlir::Operation *reductionOpUse : reductionOp->getUsers()) {
    if (auto convertReduction =
            mlir::dyn_cast<fir::ConvertOp>(reductionOpUse)) {
      for (mlir::Operation *convertReductionUse :
           convertReduction.getRes().getUsers()) {
        if (auto storeOp = mlir::dyn_cast<fir::StoreOp>(convertReductionUse)) {
          if (storeOp.getMemref() == symVal)
            storeOp.erase();
        }
        if (auto assignOp =
                mlir::dyn_cast<hlfir::AssignOp>(convertReductionUse)) {
          if (assignOp.getLhs() == symVal)
            assignOp.erase();
        }
      }
    }
  }
}

bool Fortran::lower::isOpenMPTargetConstruct(
    const Fortran::parser::OpenMPConstruct &omp) {
  llvm::omp::Directive dir = llvm::omp::Directive::OMPD_unknown;
  if (const auto *block =
          std::get_if<Fortran::parser::OpenMPBlockConstruct>(&omp.u)) {
    const auto &begin =
        std::get<Fortran::parser::OmpBeginBlockDirective>(block->t);
    dir = std::get<Fortran::parser::OmpBlockDirective>(begin.t).v;
  } else if (const auto *loop =
                 std::get_if<Fortran::parser::OpenMPLoopConstruct>(&omp.u)) {
    const auto &begin =
        std::get<Fortran::parser::OmpBeginLoopDirective>(loop->t);
    dir = std::get<Fortran::parser::OmpLoopDirective>(begin.t).v;
  }
  return llvm::omp::allTargetSet.test(dir);
}

void Fortran::lower::gatherOpenMPDeferredDeclareTargets(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclarativeConstruct &ompDecl,
    llvm::SmallVectorImpl<OMPDeferredDeclareTargetInfo>
        &deferredDeclareTarget) {
  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPDeclareTargetConstruct &ompReq) {
            collectDeferredDeclareTargets(converter, semaCtx, eval, ompReq,
                                          deferredDeclareTarget);
          },
          [&](const auto &) {},
      },
      ompDecl.u);
}

bool Fortran::lower::isOpenMPDeviceDeclareTarget(
    Fortran::lower::AbstractConverter &converter,
    Fortran::semantics::SemanticsContext &semaCtx,
    Fortran::lower::pft::Evaluation &eval,
    const Fortran::parser::OpenMPDeclarativeConstruct &ompDecl) {
  return std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::OpenMPDeclareTargetConstruct &ompReq) {
            mlir::omp::DeclareTargetDeviceType targetType =
                getDeclareTargetFunctionDevice(converter, semaCtx, eval, ompReq)
                    .value_or(mlir::omp::DeclareTargetDeviceType::host);
            return targetType != mlir::omp::DeclareTargetDeviceType::host;
          },
          [&](const auto &) { return false; },
      },
      ompDecl.u);
}

// In certain cases such as subroutine or function interfaces which declare
// but do not define or directly call the subroutine or function in the same
// module, their lowering is delayed until after the declare target construct
// itself is processed, so there symbol is not within the table.
//
// This function will also return true if we encounter any device declare
// target cases, to satisfy checking if we require the requires attributes
// on the module.
bool Fortran::lower::markOpenMPDeferredDeclareTargetFunctions(
    mlir::Operation *mod,
    llvm::SmallVectorImpl<OMPDeferredDeclareTargetInfo> &deferredDeclareTargets,
    AbstractConverter &converter) {
  bool deviceCodeFound = false;
  auto modOp = llvm::cast<mlir::ModuleOp>(mod);
  for (auto declTar : deferredDeclareTargets) {
    mlir::Operation *op = modOp.lookupSymbol(converter.mangleName(declTar.sym));

    // Due to interfaces being optionally emitted on usage in a module,
    // not finding an operation at this point cannot be a hard error, we
    // simply ignore it for now.
    // TODO: Add semantic checks for detecting cases where an erronous
    // (undefined) symbol has been supplied to a declare target clause
    if (!op)
      continue;

    auto devType = declTar.declareTargetDeviceType;
    if (!deviceCodeFound && devType != mlir::omp::DeclareTargetDeviceType::host)
      deviceCodeFound = true;

    markDeclareTarget(op, converter, declTar.declareTargetCaptureClause,
                      devType);
  }

  return deviceCodeFound;
}

void Fortran::lower::genOpenMPRequires(
    mlir::Operation *mod, const Fortran::semantics::Symbol *symbol) {
  using MlirRequires = mlir::omp::ClauseRequires;
  using SemaRequires = Fortran::semantics::WithOmpDeclarative::RequiresFlag;

  if (auto offloadMod =
          llvm::dyn_cast<mlir::omp::OffloadModuleInterface>(mod)) {
    Fortran::semantics::WithOmpDeclarative::RequiresFlags semaFlags;
    if (symbol) {
      Fortran::common::visit(
          [&](const auto &details) {
            if constexpr (std::is_base_of_v<
                              Fortran::semantics::WithOmpDeclarative,
                              std::decay_t<decltype(details)>>) {
              if (details.has_ompRequires())
                semaFlags = *details.ompRequires();
            }
          },
          symbol->details());
    }

    MlirRequires mlirFlags = MlirRequires::none;
    if (semaFlags.test(SemaRequires::ReverseOffload))
      mlirFlags = mlirFlags | MlirRequires::reverse_offload;
    if (semaFlags.test(SemaRequires::UnifiedAddress))
      mlirFlags = mlirFlags | MlirRequires::unified_address;
    if (semaFlags.test(SemaRequires::UnifiedSharedMemory))
      mlirFlags = mlirFlags | MlirRequires::unified_shared_memory;
    if (semaFlags.test(SemaRequires::DynamicAllocators))
      mlirFlags = mlirFlags | MlirRequires::dynamic_allocators;

    offloadMod.setRequires(mlirFlags);
  }
}
