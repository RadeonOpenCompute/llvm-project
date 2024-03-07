//===-- include/flang/Semantics/openmp-directive-sets.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_
#define FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_

#include "flang/Common/enum-set.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using OmpDirectiveSet = Fortran::common::EnumSet<llvm::omp::Directive,
    llvm::omp::Directive_enumSize>;

namespace llvm::omp {
//===----------------------------------------------------------------------===//
// Directive sets for single directives
//===----------------------------------------------------------------------===//
// - top<Directive>Set: The directive appears alone or as the first in a
//   combined construct.
// - all<Directive>Set: All standalone or combined uses of the directive.

static const OmpDirectiveSet topParallelSet{
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_masked_taskloop_simd,
    Directive::OMPD_parallel_masked_taskloop,
    Directive::OMPD_parallel_master_taskloop_simd,
    Directive::OMPD_parallel_master_taskloop,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
    Directive::OMPD_parallel,
};

static const OmpDirectiveSet allParallelSet{
    OmpDirectiveSet{
        Directive::OMPD_distribute_parallel_do_simd,
        Directive::OMPD_distribute_parallel_do,
        Directive::OMPD_target_parallel_do_simd,
        Directive::OMPD_target_parallel_do,
        Directive::OMPD_target_parallel,
        Directive::OMPD_target_teams_distribute_parallel_do_simd,
        Directive::OMPD_target_teams_distribute_parallel_do,
        Directive::OMPD_teams_distribute_parallel_do_simd,
        Directive::OMPD_teams_distribute_parallel_do,
    } | topParallelSet,
};

static const OmpDirectiveSet topDoSet{
    Directive::OMPD_do_simd,
    Directive::OMPD_do,
};

static const OmpDirectiveSet allDoSet{
    OmpDirectiveSet{
        Directive::OMPD_distribute_parallel_do_simd,
        Directive::OMPD_distribute_parallel_do,
        Directive::OMPD_parallel_do_simd,
        Directive::OMPD_parallel_do,
        Directive::OMPD_target_parallel_do_simd,
        Directive::OMPD_target_parallel_do,
        Directive::OMPD_target_teams_distribute_parallel_do_simd,
        Directive::OMPD_target_teams_distribute_parallel_do,
        Directive::OMPD_teams_distribute_parallel_do_simd,
        Directive::OMPD_teams_distribute_parallel_do,
    } | topDoSet,
};

static const OmpDirectiveSet topTaskloopSet{
    Directive::OMPD_taskloop_simd,
    Directive::OMPD_taskloop,
};

static const OmpDirectiveSet allTaskloopSet{
    OmpDirectiveSet{
        Directive::OMPD_masked_taskloop_simd,
        Directive::OMPD_masked_taskloop,
        Directive::OMPD_master_taskloop_simd,
        Directive::OMPD_master_taskloop,
        Directive::OMPD_parallel_masked_taskloop_simd,
        Directive::OMPD_parallel_masked_taskloop,
        Directive::OMPD_parallel_master_taskloop_simd,
        Directive::OMPD_parallel_master_taskloop,
    } | topTaskloopSet,
};

static const OmpDirectiveSet topTargetSet{
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel,
    Directive::OMPD_target_simd,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_simd,
    Directive::OMPD_target_teams_distribute,
    Directive::OMPD_target_teams,
    Directive::OMPD_target,
};

static const OmpDirectiveSet allTargetSet{topTargetSet};

static const OmpDirectiveSet topSimdSet{
    Directive::OMPD_simd,
};

static const OmpDirectiveSet allSimdSet{
    OmpDirectiveSet{
        Directive::OMPD_distribute_parallel_do_simd,
        Directive::OMPD_distribute_simd,
        Directive::OMPD_do_simd,
        Directive::OMPD_masked_taskloop_simd,
        Directive::OMPD_master_taskloop_simd,
        Directive::OMPD_parallel_do_simd,
        Directive::OMPD_parallel_masked_taskloop_simd,
        Directive::OMPD_parallel_master_taskloop_simd,
        Directive::OMPD_target_parallel_do_simd,
        Directive::OMPD_target_simd,
        Directive::OMPD_target_teams_distribute_parallel_do_simd,
        Directive::OMPD_target_teams_distribute_simd,
        Directive::OMPD_taskloop_simd,
        Directive::OMPD_teams_distribute_parallel_do_simd,
        Directive::OMPD_teams_distribute_simd,
    } | topSimdSet,
};

static const OmpDirectiveSet topTeamsSet{
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_simd,
    Directive::OMPD_teams_distribute,
    Directive::OMPD_teams,
};

static const OmpDirectiveSet allTeamsSet{
    OmpDirectiveSet{
        llvm::omp::OMPD_target_teams_distribute_parallel_do_simd,
        llvm::omp::OMPD_target_teams_distribute_parallel_do,
        llvm::omp::OMPD_target_teams_distribute_simd,
        llvm::omp::OMPD_target_teams_distribute,
        llvm::omp::OMPD_target_teams,
    } | topTeamsSet,
};

static const OmpDirectiveSet topDistributeSet{
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_simd,
    Directive::OMPD_distribute,
};

static const OmpDirectiveSet allDistributeSet{
    OmpDirectiveSet{
        llvm::omp::OMPD_target_teams_distribute_parallel_do_simd,
        llvm::omp::OMPD_target_teams_distribute_parallel_do,
        llvm::omp::OMPD_target_teams_distribute_simd,
        llvm::omp::OMPD_target_teams_distribute,
        llvm::omp::OMPD_teams_distribute_parallel_do_simd,
        llvm::omp::OMPD_teams_distribute_parallel_do,
        llvm::omp::OMPD_teams_distribute_simd,
        llvm::omp::OMPD_teams_distribute,
    } | topDistributeSet,
};

//===----------------------------------------------------------------------===//
// Directive sets for groups of multiple directives
//===----------------------------------------------------------------------===//

static const OmpDirectiveSet allDoSimdSet{allDoSet & allSimdSet};

static const OmpDirectiveSet workShareSet{
    OmpDirectiveSet{
        Directive::OMPD_workshare,
        Directive::OMPD_parallel_workshare,
        Directive::OMPD_parallel_sections,
        Directive::OMPD_sections,
        Directive::OMPD_single,
    } | allDoSet,
};

static const OmpDirectiveSet taskGeneratingSet{
    OmpDirectiveSet{
        Directive::OMPD_task,
    } | allTaskloopSet,
};

static const OmpDirectiveSet nonPartialVarSet{
    Directive::OMPD_allocate,
    Directive::OMPD_allocators,
    Directive::OMPD_threadprivate,
    Directive::OMPD_declare_target,
};

static const OmpDirectiveSet loopConstructSet{
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_simd,
    Directive::OMPD_distribute,
    Directive::OMPD_do_simd,
    Directive::OMPD_do,
    Directive::OMPD_masked_taskloop,
    Directive::OMPD_masked_taskloop_simd,
    Directive::OMPD_master_taskloop,
    Directive::OMPD_master_taskloop_simd,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_masked_taskloop,
    Directive::OMPD_parallel_masked_taskloop_simd,
    Directive::OMPD_parallel_master_taskloop,
    Directive::OMPD_parallel_master_taskloop_simd,
    Directive::OMPD_simd,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_simd,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_simd,
    Directive::OMPD_target_teams_distribute,
    Directive::OMPD_taskloop_simd,
    Directive::OMPD_taskloop,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_simd,
    Directive::OMPD_teams_distribute,
    Directive::OMPD_tile,
    Directive::OMPD_unroll,
};

static const OmpDirectiveSet blockConstructSet{
    Directive::OMPD_master,
    Directive::OMPD_ordered,
    Directive::OMPD_parallel_workshare,
    Directive::OMPD_parallel,
    Directive::OMPD_single,
    Directive::OMPD_target_data,
    Directive::OMPD_target_parallel,
    Directive::OMPD_target_teams,
    Directive::OMPD_target,
    Directive::OMPD_task,
    Directive::OMPD_taskgroup,
    Directive::OMPD_teams,
    Directive::OMPD_workshare,
};

//===----------------------------------------------------------------------===//
// Directive sets for allowed/not allowed nested directives
//===----------------------------------------------------------------------===//

static const OmpDirectiveSet nestedOrderedErrSet{
    Directive::OMPD_critical,
    Directive::OMPD_ordered,
    Directive::OMPD_atomic,
    Directive::OMPD_task,
    Directive::OMPD_taskloop,
};

static const OmpDirectiveSet nestedWorkshareErrSet{
    OmpDirectiveSet{
        Directive::OMPD_task,
        Directive::OMPD_taskloop,
        Directive::OMPD_critical,
        Directive::OMPD_ordered,
        Directive::OMPD_atomic,
        Directive::OMPD_master,
    } | workShareSet,
};

static const OmpDirectiveSet nestedMasterErrSet{
    OmpDirectiveSet{
        Directive::OMPD_atomic,
    } | taskGeneratingSet |
        workShareSet,
};

static const OmpDirectiveSet nestedBarrierErrSet{
    OmpDirectiveSet{
        Directive::OMPD_critical,
        Directive::OMPD_ordered,
        Directive::OMPD_atomic,
        Directive::OMPD_master,
    } | taskGeneratingSet |
        workShareSet,
};

static const OmpDirectiveSet nestedTeamsAllowedSet{
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_master,
    Directive::OMPD_parallel_master_taskloop,
    Directive::OMPD_parallel_master_taskloop_simd,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
    Directive::OMPD_distribute,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
};

static const OmpDirectiveSet nestedOrderedParallelErrSet{
    Directive::OMPD_parallel,
    Directive::OMPD_target_parallel,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
};

static const OmpDirectiveSet nestedOrderedDoAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_parallel_do,
    Directive::OMPD_target_parallel_do,
};

static const OmpDirectiveSet nestedCancelTaskgroupAllowedSet{
    Directive::OMPD_task,
    Directive::OMPD_taskloop,
};

static const OmpDirectiveSet nestedCancelSectionsAllowedSet{
    Directive::OMPD_sections,
    Directive::OMPD_parallel_sections,
};

static const OmpDirectiveSet nestedCancelDoAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_parallel_do,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do,
};

static const OmpDirectiveSet nestedCancelParallelAllowedSet{
    Directive::OMPD_parallel,
    Directive::OMPD_target_parallel,
};

static const OmpDirectiveSet nestedReduceWorkshareAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_sections,
    Directive::OMPD_do_simd,
};
} // namespace llvm::omp

#endif // FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_
