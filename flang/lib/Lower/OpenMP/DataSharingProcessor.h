//===-- Lower/OpenMP/DataSharingProcessor.h ---------------------*- C++ -*-===//
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
#ifndef FORTRAN_LOWER_DATASHARINGPROCESSOR_H
#define FORTRAN_LOWER_DATASHARINGPROCESSOR_H

#include "Clauses.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/OpenMP.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/symbol.h"

namespace mlir {
namespace omp {
struct PrivateClauseOps;
} // namespace omp
} // namespace mlir

namespace Fortran {
namespace lower {
namespace omp {

class DataSharingProcessor {
private:
  bool hasLastPrivateOp;
  mlir::OpBuilder::InsertPoint lastPrivIP;
  mlir::OpBuilder::InsertPoint insPt;
  mlir::Value loopIV;
  // Symbols in private, firstprivate, and/or lastprivate clauses.
  llvm::SetVector<const Fortran::semantics::Symbol *> privatizedSymbols;
  llvm::SetVector<const Fortran::semantics::Symbol *> defaultSymbols;
  llvm::SetVector<const Fortran::semantics::Symbol *> symbolsInNestedRegions;
  llvm::SetVector<const Fortran::semantics::Symbol *> symbolsInParentRegions;
  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &firOpBuilder;
  omp::List<omp::Clause> clauses;
  Fortran::lower::pft::Evaluation &eval;
  bool privatizationDone = false;

  bool useDelayedPrivatization;
  Fortran::lower::SymMap *symTable;

  bool needBarrier();
  void collectSymbols(Fortran::semantics::Symbol::Flag flag);
  void collectOmpObjectListSymbol(
      const omp::ObjectList &objects,
      llvm::SetVector<const Fortran::semantics::Symbol *> &symbolSet);
  void collectSymbolsForPrivatization();
  void insertBarrier();
  void collectDefaultSymbols();
  void privatize(
      mlir::omp::PrivateClauseOps *clauseOps,
      llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *privateSyms);
  void defaultPrivatize(
      mlir::omp::PrivateClauseOps *clauseOps,
      llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *privateSyms);
  void doPrivatize(
      const Fortran::semantics::Symbol *sym,
      mlir::omp::PrivateClauseOps *clauseOps,
      llvm::SmallVectorImpl<const Fortran::semantics::Symbol *> *privateSyms);
  void copyLastPrivatize(mlir::Operation *op);
  void insertLastPrivateCompare(mlir::Operation *op);
  void cloneSymbol(const Fortran::semantics::Symbol *sym);
  void
  copyFirstPrivateSymbol(const Fortran::semantics::Symbol *sym,
                         mlir::OpBuilder::InsertPoint *copyAssignIP = nullptr);
  void copyLastPrivateSymbol(const Fortran::semantics::Symbol *sym,
                             mlir::OpBuilder::InsertPoint *lastPrivIP);
  void insertDeallocs();

public:
  DataSharingProcessor(Fortran::lower::AbstractConverter &converter,
                       Fortran::semantics::SemanticsContext &semaCtx,
                       const List<Clause> &clauses,
                       Fortran::lower::pft::Evaluation &eval,
                       bool useDelayedPrivatization = false,
                       Fortran::lower::SymMap *symTable = nullptr)
      : hasLastPrivateOp(false), converter(converter),
        firOpBuilder(converter.getFirOpBuilder()), clauses(clauses), eval(eval),
        useDelayedPrivatization(useDelayedPrivatization), symTable(symTable) {}

  // Privatisation is split into 3 steps:
  //
  // * Step1: collects all symbols that should be privatized.
  //
  // * Step2: performs cloning of all privatisation clauses and copying for
  // firstprivates. Step2 is performed at the place where process/processStep2
  // is called. This is usually inside the Operation corresponding to the OpenMP
  // construct, for looping constructs this is just before the Operation.
  //
  // * Step3: performs the copying for lastprivates and requires knowledge of
  // the MLIR operation to insert the last private update. Step3 adds
  // dealocation code as well.
  //
  // The split was performed for the following reasons:
  //
  // 1. Step1 was split so that the `target` op knows which symbols should not
  // be mapped into the target region due to being `private`. The implicit
  // mapping happens before the op body is generated so we need to to collect
  // the private symbols first and then later in the body actually privatize
  // them.
  //
  // 2. Step2 was split in order to call privatisation for looping constructs
  // before the operation is created since the bounds of the MLIR OpenMP
  // operation can be privatised.
  void processStep1();
  void processStep2(mlir::omp::PrivateClauseOps *clauseOps = nullptr,
                    llvm::SmallVectorImpl<const Fortran::semantics::Symbol *>
                        *privateSyms = nullptr);
  void processStep3(mlir::Operation *op, bool isLoop);

  void setLoopIV(mlir::Value iv) {
    assert(!loopIV && "Loop iteration variable already set");
    loopIV = iv;
  }

  const llvm::SetVector<const Fortran::semantics::Symbol *> &
  getPrivatizedSymbols() const {
    return privatizedSymbols;
  }
};

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_DATASHARINGPROCESSOR_H
