! This test checks lowering of OpenMP combined loop constructs.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-fir %s -o - | FileCheck %s

program main
  integer :: i

  ! TODO TASKLOOP SIMD

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  !$omp teams

  ! CHECK: omp.distribute
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  !$omp distribute parallel do simd
  do i = 1, 10
  end do
  !$omp end distribute parallel do simd

  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------
  !$omp teams
  
  ! CHECK: omp.distribute
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  !$omp distribute parallel do
  do i = 1, 10
  end do
  !$omp end distribute parallel do

  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------
  !$omp teams

  ! CHECK: omp.distribute
  ! CHECK: omp.simd
  !$omp distribute simd
  do i = 1, 10
  end do
  !$omp end distribute simd

  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.wsloop
  !$omp do simd
  do i = 1, 10
  end do
  !$omp end do simd

  ! ----------------------------------------------------------------------------
  ! PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  !$omp parallel do simd
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! ----------------------------------------------------------------------------
  ! PARALLEL DO
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  !$omp parallel do
  do i = 1, 10
  end do
  !$omp end parallel do

  ! ----------------------------------------------------------------------------
  ! TARGET PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.target
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  !$omp target parallel do simd
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  ! ----------------------------------------------------------------------------
  ! TARGET PARALLEL DO
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.target
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  !$omp target parallel do
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------

  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  !$omp target teams distribute parallel do simd
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------

  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  !$omp target teams distribute parallel do
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------

  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK: omp.simd
  !$omp target teams distribute simd
  do i = 1, 10
  end do
  !$omp end target teams distribute simd

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE
  ! ----------------------------------------------------------------------------

  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  !$omp target teams distribute
  do i = 1, 10
  end do
  !$omp end target teams distribute

  ! ----------------------------------------------------------------------------
  ! TARGET SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.target
  ! CHECK: omp.simd
  !$omp target simd
  do i = 1, 10
  end do
  !$omp end target simd

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------

  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  !$omp teams distribute parallel do simd
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do simd

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------

  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  !$omp teams distribute parallel do
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------

  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK: omp.simd
  !$omp teams distribute simd
  do i = 1, 10
  end do
  !$omp end teams distribute simd

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE
  ! ----------------------------------------------------------------------------

  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  !$omp teams distribute
  do i = 1, 10
  end do
  !$omp end teams distribute

end program main
