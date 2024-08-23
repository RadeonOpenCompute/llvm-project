! RUN: bbc -emit-fir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp -o - %s | FileCheck %s

! CHECK:       omp.teams
! CHECK-SAME:  reduction
subroutine reduction_teams()
  integer :: i
  i = 0

  !$omp teams reduction(+:i)
  i = i + 1
  !$omp end teams
end subroutine reduction_teams
