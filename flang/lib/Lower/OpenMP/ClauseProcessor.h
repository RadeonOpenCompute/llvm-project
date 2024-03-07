//===-- Lower/OpenMP/ClauseProcessor.h --------------------------*- C++ -*-===//
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
#ifndef FORTRAN_LOWER_CLAUASEPROCESSOR_H
#define FORTRAN_LOWER_CLAUASEPROCESSOR_H

#include "DirectivesCommon.h"
#include "OperationClauses.h"
#include "ReductionProcessor.h"
#include "Utils.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/Bridge.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parse-tree.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace Fortran {
namespace lower {
namespace omp {

/// Class that handles the processing of OpenMP clauses.
///
/// Its `process<ClauseName>()` methods perform MLIR code generation for their
/// corresponding clause if it is present in the clause list. Otherwise, they
/// will return `false` to signal that the clause was not found.
///
/// The intended use is of this class is to move clause processing outside of
/// construct processing, since the same clauses can appear attached to
/// different constructs and constructs can be combined, so that code
/// duplication is minimized.
///
/// Each construct-lowering function only calls the `process<ClauseName>()`
/// methods that relate to clauses that can impact the lowering of that
/// construct.
class ClauseProcessor {
  using ClauseTy = Fortran::parser::OmpClause;

public:
  ClauseProcessor(Fortran::lower::AbstractConverter &converter,
                  Fortran::semantics::SemanticsContext &semaCtx,
                  const Fortran::parser::OmpClauseList &clauses)
      : converter(converter), semaCtx(semaCtx), clauses(clauses) {}

  // 'Unique' clauses: They can appear at most once in the clause list.
  bool processCollapse(mlir::Location currentLocation,
                       Fortran::lower::pft::Evaluation &eval,
                       CollapseClauseOps &result) const;
  bool processDefault() const;
  bool processDevice(Fortran::lower::StatementContext &stmtCtx,
                     DeviceClauseOps &result) const;
  bool processDeviceType(DeviceTypeClauseOps &result) const;
  bool processFinal(Fortran::lower::StatementContext &stmtCtx,
                    FinalClauseOps &result) const;
  bool processHint(HintClauseOps &result) const;
  bool processMergeable(MergeableClauseOps &result) const;
  bool processNowait(NowaitClauseOps &result) const;
  bool processNumTeams(Fortran::lower::StatementContext &stmtCtx,
                       NumTeamsClauseOps &result) const;
  bool processNumThreads(Fortran::lower::StatementContext &stmtCtx,
                         NumThreadsClauseOps &result) const;
  bool processOrdered(OrderedClauseOps &result) const;
  bool processPriority(Fortran::lower::StatementContext &stmtCtx,
                       PriorityClauseOps &result) const;
  bool processProcBind(ProcBindClauseOps &result) const;
  bool processSafelen(SafelenClauseOps &result) const;
  bool processSchedule(Fortran::lower::StatementContext &stmtCtx,
                       ScheduleClauseOps &result) const;
  bool processSimdlen(SimdlenClauseOps &result) const;
  bool processThreadLimit(Fortran::lower::StatementContext &stmtCtx,
                          ThreadLimitClauseOps &result) const;
  bool processUntied(UntiedClauseOps &result) const;

  // 'Repeatable' clauses: They can appear multiple times in the clause list.
  bool processAllocate(AllocateClauseOps &result) const;
  bool processCopyin(CopyinClauseOps &result) const;
  bool processCopyprivate(mlir::Location currentLocation,
                          CopyprivateClauseOps &result) const;
  bool processDepend(DependClauseOps &result) const;
  bool processEnter(EnterLinkToClauseOps &result) const;
  bool
  processIf(Fortran::parser::OmpIfClause::DirectiveNameModifier directiveName,
            IfClauseOps &result) const;
  bool processLink(EnterLinkToClauseOps &result) const;

  // This method is used to process a map clause.
  bool processMap(mlir::Location currentLocation,
                  Fortran::lower::StatementContext &stmtCtx,
                  MapClauseOps &result) const;
  bool processReduction(mlir::Location currentLocation,
                        ReductionClauseOps &result) const;
  bool processTargetReduction(TargetReductionClauseOps &result) const;
  bool processSectionsReduction(mlir::Location currentLocation,
                                ReductionClauseOps &result) const;
  bool processTo(EnterLinkToClauseOps &result) const;
  bool processUseDeviceAddr(UseDeviceClauseOps &result) const;
  bool processUseDevicePtr(UseDeviceClauseOps &result) const;

  template <typename T>
  bool processMotionClauses(Fortran::lower::StatementContext &stmtCtx,
                            MapClauseOps &result);

  // Call this method for these clauses that should be supported but are not
  // implemented yet. It triggers a compilation error if any of the given
  // clauses is found.
  template <typename... Ts>
  void processTODO(mlir::Location currentLocation,
                   llvm::omp::Directive directive) const;

private:
  using ClauseIterator = std::list<ClauseTy>::const_iterator;

  /// Utility to find a clause within a range in the clause list.
  template <typename T>
  static ClauseIterator findClause(ClauseIterator begin, ClauseIterator end);

  /// Return the first instance of the given clause found in the clause list or
  /// `nullptr` if not present. If more than one instance is expected, use
  /// `findRepeatableClause` instead.
  template <typename T>
  const T *
  findUniqueClause(const Fortran::parser::CharBlock **source = nullptr) const;

  /// Call `callbackFn` for each occurrence of the given clause. Return `true`
  /// if at least one instance was found.
  template <typename T>
  bool findRepeatableClause(
      std::function<void(const T *, const Fortran::parser::CharBlock &source)>
          callbackFn) const;

  /// Set the `result` to a new `mlir::UnitAttr` if the clause is present.
  template <typename T>
  bool markClauseOccurrence(mlir::UnitAttr &result) const;

  Fortran::lower::AbstractConverter &converter;
  Fortran::semantics::SemanticsContext &semaCtx;
  const Fortran::parser::OmpClauseList &clauses;
};

template <typename T>
bool ClauseProcessor::processMotionClauses(
    Fortran::lower::StatementContext &stmtCtx, MapClauseOps &result) {
  return findRepeatableClause<T>(
      [&](const T *motionClause, const Fortran::parser::CharBlock &source) {
        mlir::Location clauseLocation = converter.genLocation(source);
        fir::FirOpBuilder &firOpBuilder = converter.getFirOpBuilder();

        static_assert(std::is_same_v<T, ClauseProcessor::ClauseTy::To> ||
                      std::is_same_v<T, ClauseProcessor::ClauseTy::From>);

        // TODO Support motion modifiers: present, mapper, iterator.
        constexpr llvm::omp::OpenMPOffloadMappingFlags mapTypeBits =
            std::is_same_v<T, ClauseProcessor::ClauseTy::To>
                ? llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_TO
                : llvm::omp::OpenMPOffloadMappingFlags::OMP_MAP_FROM;

        for (const Fortran::parser::OmpObject &ompObject : motionClause->v.v) {
          llvm::SmallVector<mlir::Value> bounds;
          std::stringstream asFortran;
          Fortran::lower::AddrAndBoundsInfo info =
              Fortran::lower::gatherDataOperandAddrAndBounds<
                  Fortran::parser::OmpObject, mlir::omp::DataBoundsOp,
                  mlir::omp::DataBoundsType>(
                  converter, firOpBuilder, semaCtx, stmtCtx, ompObject,
                  clauseLocation, asFortran, bounds, treatIndexAsSection);

          auto origSymbol =
              converter.getSymbolAddress(*getOmpObjectSymbol(ompObject));
          mlir::Value symAddr = info.addr;
          if (origSymbol && fir::isTypeWithDescriptor(origSymbol.getType()))
            symAddr = origSymbol;

          // Explicit map captures are captured ByRef by default,
          // optimisation passes may alter this to ByCopy or other capture
          // types to optimise
          mlir::Value mapOp = createMapInfoOp(
              firOpBuilder, clauseLocation, symAddr, mlir::Value{},
              asFortran.str(), bounds, {},
              static_cast<
                  std::underlying_type_t<llvm::omp::OpenMPOffloadMappingFlags>>(
                  mapTypeBits),
              mlir::omp::VariableCaptureKind::ByRef, symAddr.getType());

          result.mapVars.push_back(mapOp);
        }
      });
}

template <typename... Ts>
void ClauseProcessor::processTODO(mlir::Location currentLocation,
                                  llvm::omp::Directive directive) const {
  auto checkUnhandledClause = [&](const auto *x) {
    if (!x)
      return;
    TODO(currentLocation,
         "Unhandled clause " +
             llvm::StringRef(Fortran::parser::ParseTreeDumper::GetNodeName(*x))
                 .upper() +
             " in " + llvm::omp::getOpenMPDirectiveName(directive).upper() +
             " construct");
  };

  for (ClauseIterator it = clauses.v.begin(); it != clauses.v.end(); ++it)
    (checkUnhandledClause(std::get_if<Ts>(&it->u)), ...);
}

template <typename T>
ClauseProcessor::ClauseIterator
ClauseProcessor::findClause(ClauseIterator begin, ClauseIterator end) {
  for (ClauseIterator it = begin; it != end; ++it) {
    if (std::get_if<T>(&it->u))
      return it;
  }

  return end;
}

template <typename T>
const T *ClauseProcessor::findUniqueClause(
    const Fortran::parser::CharBlock **source) const {
  ClauseIterator it = findClause<T>(clauses.v.begin(), clauses.v.end());
  if (it != clauses.v.end()) {
    if (source)
      *source = &it->source;
    return &std::get<T>(it->u);
  }
  return nullptr;
}

template <typename T>
bool ClauseProcessor::findRepeatableClause(
    std::function<void(const T *, const Fortran::parser::CharBlock &source)>
        callbackFn) const {
  bool found = false;
  ClauseIterator nextIt, endIt = clauses.v.end();
  for (ClauseIterator it = clauses.v.begin(); it != endIt; it = nextIt) {
    nextIt = findClause<T>(it, endIt);

    if (nextIt != endIt) {
      callbackFn(&std::get<T>(nextIt->u), nextIt->source);
      found = true;
      ++nextIt;
    }
  }
  return found;
}

template <typename T>
bool ClauseProcessor::markClauseOccurrence(mlir::UnitAttr &result) const {
  if (findUniqueClause<T>()) {
    result = converter.getFirOpBuilder().getUnitAttr();
    return true;
  }
  return false;
}

} // namespace omp
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_CLAUASEPROCESSOR_H
