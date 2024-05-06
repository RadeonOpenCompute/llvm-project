! This test checks lowering of OpenMP do simd safelen() pragma

! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s
subroutine testDoSimdSafelen(int_array)
  integer :: int_array(*)

  !CHECK: omp.wsloop {
  !CHECK: omp.simd safelen(4) {
  !CHECK: omp.loop_nest {{.*}} {
  !$omp do simd safelen(4)
    do index_ = 1, 10
    end do
  !$omp end do simd

end subroutine testDoSimdSafelen
