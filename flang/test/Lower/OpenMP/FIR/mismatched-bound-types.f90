! RUN: %flang_fc1 -fopenmp -emit-fir %s -o - | FileCheck %s

! Check that this testcase is lowered to FIR successfully.
! CHECK: omp.target trip_count

module Test
    use, intrinsic :: ISO_Fortran_env, only: REAL64,INT64
    implicit none
    integer(kind=INT64) :: N
    real(kind=REAL64), allocatable :: A(:)

    contains
        subroutine init_arrays(initA)
            implicit none
            real(kind=REAL64), intent(in) :: initA
            integer(kind=INT64) :: i
            !$omp target teams distribute parallel do
            do i = 1, N
                A(i) = initA
            end do
        end subroutine init_arrays

end module Test
