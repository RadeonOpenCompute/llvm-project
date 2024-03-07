//===-- Lower/OpenMP/OperationClauses.h -------------------------*- C++ -*-===//
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

#include "Utils.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include <optional>

namespace Fortran {
namespace semantics {
class Symbol;
} // namespace semantics
} // namespace Fortran

namespace Fortran {
namespace lower {
namespace omp {

//===----------------------------------------------------------------------===//
// Mixin structures defining operands associated with each OpenMP clause.
//===----------------------------------------------------------------------===//

struct AlignedClauseOps {
  llvm::SmallVector<mlir::Value> alignedVars;
  llvm::SmallVector<mlir::Attribute> alignmentAttrs;
};

struct AllocateClauseOps {
  llvm::SmallVector<mlir::Value> allocatorVars, allocateVars;
};

struct CollapseClauseOps {
  llvm::SmallVector<mlir::Value> loopLBVar, loopUBVar, loopStepVar;
  llvm::SmallVector<const Fortran::semantics::Symbol *> loopIV;
};

struct CopyinClauseOps {};

struct CopyprivateClauseOps {
  llvm::SmallVector<mlir::Value> copyprivateVars;
  llvm::SmallVector<mlir::Attribute> copyprivateFuncs;
};

struct DependClauseOps {
  llvm::SmallVector<mlir::Attribute> dependTypeAttrs;
  llvm::SmallVector<mlir::Value> dependVars;
};

struct DeviceClauseOps {
  mlir::Value deviceVar;
};

struct DeviceTypeClauseOps {
  mlir::omp::DeclareTargetDeviceType deviceType;
};

struct DistScheduleClauseOps {
  mlir::UnitAttr distScheduleStaticAttr;
  mlir::Value distScheduleChunkSizeVar;
};

struct EnterLinkToClauseOps {
  llvm::SmallVector<DeclareTargetCapturePair> symbolAndClause;
};

struct FinalClauseOps {
  mlir::Value finalVar;
};

struct GrainsizeClauseOps {
  mlir::Value grainsizeVar;
};

struct HintClauseOps {
  mlir::IntegerAttr hintAttr;
};

struct IfClauseOps {
  mlir::Value ifVar;
};

struct InReductionClauseOps {
  llvm::SmallVector<mlir::Value> inReductionVars;
  llvm::SmallVector<mlir::Type> inReductionTypes;
  llvm::SmallVector<mlir::Attribute> inReductionDeclSymbols;
  std::optional<llvm::SmallVector<const Fortran::semantics::Symbol *>>
      inReductionSymbols;
};

struct LinearClauseOps {
  llvm::SmallVector<mlir::Value> linearVars, linearStepVars;
};

// The optional parameters - mapSymTypes, mapSymLocs & mapSymbols are used to
// store the original type, location and Fortran symbol for the map operands.
// They may be used later on to create the block_arguments for some of the
// target directives that require it.
struct MapClauseOps {
  llvm::SmallVector<mlir::Value> mapVars;
  std::optional<llvm::SmallVector<mlir::Type>> mapSymTypes;
  std::optional<llvm::SmallVector<mlir::Location>> mapSymLocs;
  std::optional<llvm::SmallVector<const Fortran::semantics::Symbol *>>
      mapSymbols;
};

struct MergeableClauseOps {
  mlir::UnitAttr mergeableAttr;
};

struct NogroupClauseOps {
  mlir::UnitAttr nogroupAttr;
};

struct NontemporalClauseOps {
  llvm::SmallVector<mlir::Value> nontemporalVars;
};

struct NowaitClauseOps {
  mlir::UnitAttr nowaitAttr;
};

struct NumTasksClauseOps {
  mlir::Value numTasksVar;
};

struct NumTeamsClauseOps {
  mlir::Value numTeamsLowerVar;
  mlir::Value numTeamsUpperVar;
};

struct NumThreadsClauseOps {
  mlir::Value numThreadsVar;
};

struct OrderClauseOps {
  mlir::omp::ClauseOrderKindAttr orderAttr;
};

struct OrderedClauseOps {
  mlir::IntegerAttr orderedAttr;
};

struct ParallelizationLevelClauseOps {
  mlir::UnitAttr parLevelThreadsAttr;
  mlir::UnitAttr parLevelSimdAttr;
};

struct PriorityClauseOps {
  mlir::Value priorityVar;
};

struct PrivateClauseOps {
  llvm::SmallVector<mlir::Value> privateVars;
  llvm::SmallVector<mlir::Attribute> privatizers;
};

struct ProcBindClauseOps {
  mlir::omp::ClauseProcBindKindAttr procBindKindAttr;
};

struct ReductionClauseOps {
  llvm::SmallVector<mlir::Value> reductionVars;
  llvm::SmallVector<mlir::Type> reductionTypes;
  llvm::SmallVector<mlir::Attribute> reductionDeclSymbols;
  std::optional<llvm::SmallVector<const Fortran::semantics::Symbol *>>
      reductionSymbols;
};

struct SafelenClauseOps {
  mlir::IntegerAttr safelenAttr;
};

struct ScheduleClauseOps {
  mlir::omp::ClauseScheduleKindAttr scheduleValAttr;
  mlir::omp::ScheduleModifierAttr scheduleModAttr;
  mlir::Value scheduleChunkVar;
  mlir::UnitAttr scheduleSimdAttr;
};

struct SimdlenClauseOps {
  mlir::IntegerAttr simdlenAttr;
};

struct TargetReductionClauseOps {
  llvm::SmallVector<const Fortran::semantics::Symbol *> targetReductionSymbols;
};

struct TaskReductionClauseOps {
  llvm::SmallVector<mlir::Value> taskReductionVars;
  llvm::SmallVector<mlir::Type> taskReductionTypes;
  llvm::SmallVector<mlir::Attribute> taskReductionDeclSymbols;
  std::optional<llvm::SmallVector<const Fortran::semantics::Symbol *>>
      taskReductionSymbols;
};

struct ThreadLimitClauseOps {
  mlir::Value threadLimitVar;
};

struct UntiedClauseOps {
  mlir::UnitAttr untiedAttr;
};

struct UseDeviceClauseOps {
  llvm::SmallVector<mlir::Value> useDevicePtrVars;
  llvm::SmallVector<mlir::Value> useDeviceAddrVars;
  llvm::SmallVector<mlir::Type> useDeviceTypes;
  llvm::SmallVector<mlir::Location> useDeviceLocs;
  llvm::SmallVector<const Fortran::semantics::Symbol *> useDeviceSymbols;
};

//===----------------------------------------------------------------------===//
// Structures defining clause operands associated with each OpenMP leaf
// construct.
//
// These mirror the arguments expected by the corresponding OpenMP MLIR ops.
//===----------------------------------------------------------------------===//

namespace detail {
template <typename... Mixins>
struct Clauses : public Mixins... {};
} // namespace detail

using CriticalDeclareOpClauseOps = detail::Clauses<HintClauseOps>;

using DataOpClauseOps = detail::Clauses<DeviceClauseOps, IfClauseOps,
                                        MapClauseOps, UseDeviceClauseOps>;

using DeclareTargetOpClauseOps = detail::Clauses<EnterLinkToClauseOps>;

using DistributeOpClauseOps =
    detail::Clauses<AllocateClauseOps, DistScheduleClauseOps, OrderClauseOps>;

using EnterExitUpdateDataOpClauseOps =
    detail::Clauses<DependClauseOps, DeviceClauseOps, IfClauseOps, MapClauseOps,
                    NowaitClauseOps>;

using LoopNestOpClauseOps = detail::Clauses<CollapseClauseOps>;

// TODO Rename to "masked"
// TODO `filter` clause.
using MasterOpClauseOps = detail::Clauses<>;

using OrderedRegionOpClauseOps = detail::Clauses<ParallelizationLevelClauseOps>;

using ParallelOpClauseOps =
    detail::Clauses<AllocateClauseOps, IfClauseOps, NumThreadsClauseOps,
                    PrivateClauseOps, ProcBindClauseOps, ReductionClauseOps>;

using SectionsOpClauseOps =
    detail::Clauses<AllocateClauseOps, NowaitClauseOps, ReductionClauseOps>;

// TODO `linear` clause.
using SimdLoopOpClauseOps =
    detail::Clauses<AlignedClauseOps, IfClauseOps, NontemporalClauseOps,
                    OrderClauseOps, ReductionClauseOps, SafelenClauseOps,
                    SimdlenClauseOps>;

using SingleOpClauseOps =
    detail::Clauses<AllocateClauseOps, CopyprivateClauseOps, NowaitClauseOps>;

// TODO `allocate`, `defaultmap`, `has_device_addr`, `in_reduction`,
// `is_device_ptr`, `uses_allocators` clauses.
using TargetOpClauseOps =
    detail::Clauses<DependClauseOps, DeviceClauseOps, IfClauseOps, MapClauseOps,
                    NowaitClauseOps, TargetReductionClauseOps,
                    ThreadLimitClauseOps>;

using TaskGroupOpClauseOps =
    detail::Clauses<AllocateClauseOps, TaskReductionClauseOps>;

using TaskLoopOpClauseOps =
    detail::Clauses<AllocateClauseOps, FinalClauseOps, GrainsizeClauseOps,
                    IfClauseOps, InReductionClauseOps, MergeableClauseOps,
                    NogroupClauseOps, NumTasksClauseOps, PriorityClauseOps,
                    ReductionClauseOps, UntiedClauseOps>;

// TODO `affinity`, `detach` clauses.
using TaskOpClauseOps =
    detail::Clauses<AllocateClauseOps, DependClauseOps, FinalClauseOps,
                    IfClauseOps, InReductionClauseOps, MergeableClauseOps,
                    PriorityClauseOps, UntiedClauseOps>;

// TODO `depend`, `nowait` clauses.
using TaskWaitOpClauseOps = detail::Clauses<>;

using TeamsOpClauseOps =
    detail::Clauses<AllocateClauseOps, IfClauseOps, NumTeamsClauseOps,
                    ReductionClauseOps, ThreadLimitClauseOps>;

// TODO `allocate` clause.
using WsloopOpClauseOps =
    detail::Clauses<LinearClauseOps, NowaitClauseOps, OrderClauseOps,
                    OrderedClauseOps, ReductionClauseOps, ScheduleClauseOps>;

} // namespace omp
} // namespace lower
} // namespace Fortran
