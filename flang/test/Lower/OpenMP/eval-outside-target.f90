! The "thread_limit" clause was added to the "target" construct in OpenMP 5.1.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefixes=BOTH,HOST
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefixes=BOTH,DEVICE

! CHECK-LABEL: func.func @_QPteams
subroutine teams()
  ! BOTH: omp.target

  ! HOST-SAME: num_teams({{.*}})
  ! HOST-SAME: teams_thread_limit({{.*}})

  ! DEVICE-NOT: num_teams({{.*}})
  ! DEVICE-NOT: teams_thread_limit({{.*}})
  ! DEVICE-SAME: {
  !$omp target

  ! BOTH: omp.teams

  ! HOST-NOT: num_teams({{.*}})
  ! HOST-NOT: thread_limit({{.*}})
  ! HOST-SAME: {

  ! DEVICE-SAME: num_teams({{.*}})
  ! DEVICE-SAME: thread_limit({{.*}})
  !$omp teams num_teams(1) thread_limit(2)
  call foo()
  !$omp end teams

  !$omp end target

  ! BOTH: omp.teams
  ! BOTH-SAME: num_teams({{.*}})
  ! BOTH-SAME: thread_limit({{.*}})
  !$omp teams num_teams(1) thread_limit(2)
  call foo()
  !$omp end teams
end subroutine teams

subroutine parallel()
  ! BOTH: omp.target

  ! HOST-SAME: num_threads({{.*}})

  ! DEVICE-NOT: num_threads({{.*}})
  ! DEVICE-SAME: {
  !$omp target

  ! BOTH: omp.parallel

  ! HOST-NOT: num_threads({{.*}})
  ! HOST-SAME: {
  
  ! DEVICE-SAME: num_threads({{.*}})
  !$omp parallel num_threads(1)
  call foo()
  !$omp end parallel
  !$omp end target

  ! BOTH: omp.target
  ! BOTH-NOT: num_threads({{.*}})
  ! BOTH-SAME: {
  !$omp target
  call foo()

  ! BOTH: omp.parallel
  ! BOTH-SAME: num_threads({{.*}})
  !$omp parallel num_threads(1)
  call foo()
  !$omp end parallel
  !$omp end target

  ! BOTH: omp.parallel
  ! BOTH-SAME: num_threads({{.*}})
  !$omp parallel num_threads(1)
  call foo()
  !$omp end parallel
end subroutine parallel

subroutine distribute_parallel_do()
  ! BOTH: omp.target
  
  ! HOST-SAME: num_threads({{.*}})
  
  ! DEVICE-NOT: num_threads({{.*}})
  ! DEVICE-SAME: {

  ! BOTH: omp.teams
  !$omp target teams

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.parallel

  ! HOST-NOT: num_threads({{.*}})
  ! HOST-SAME: {
  
  ! DEVICE-SAME: num_threads({{.*}})
  
  ! BOTH-NEXT: omp.wsloop
  !$omp distribute parallel do num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do
  !$omp end target teams

  ! BOTH: omp.target
  ! BOTH-NOT: num_threads({{.*}})
  ! BOTH-SAME: {
  ! BOTH: omp.teams
  !$omp target teams
  call foo()

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.parallel
  ! BOTH-SAME: num_threads({{.*}})
  ! BOTH-NEXT: omp.wsloop
  !$omp distribute parallel do num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do
  !$omp end target teams

  ! BOTH: omp.teams
  !$omp teams

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.parallel
  ! BOTH-SAME: num_threads({{.*}})
  ! BOTH-NEXT: omp.wsloop
  !$omp distribute parallel do num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do
  !$omp end teams
end subroutine distribute_parallel_do

subroutine distribute_parallel_do_simd()
  ! BOTH: omp.target
  
  ! HOST-SAME: num_threads({{.*}})
  
  ! DEVICE-NOT: num_threads({{.*}})
  ! DEVICE-SAME: {

  ! BOTH: omp.teams
  !$omp target teams

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.parallel

  ! HOST-NOT: num_threads({{.*}})
  ! HOST-SAME: {
  
  ! DEVICE-SAME: num_threads({{.*}})
  
  ! BOTH-NEXT: omp.wsloop
  ! BOTH-NEXT: omp.simd
  !$omp distribute parallel do simd num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do simd
  !$omp end target teams

  ! BOTH: omp.target
  ! BOTH-NOT: num_threads({{.*}})
  ! BOTH-SAME: {
  ! BOTH: omp.teams
  !$omp target teams
  call foo()

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.parallel
  ! BOTH-SAME: num_threads({{.*}})
  ! BOTH-NEXT: omp.wsloop
  ! BOTH-NEXT: omp.simd
  !$omp distribute parallel do simd num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do simd
  !$omp end target teams

  ! BOTH: omp.teams
  !$omp teams

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.parallel
  ! BOTH-SAME: num_threads({{.*}})
  ! BOTH-NEXT: omp.wsloop
  ! BOTH-NEXT: omp.simd
  !$omp distribute parallel do simd num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do simd
  !$omp end teams
end subroutine distribute_parallel_do_simd
