! This test checks the insertion of lifetime information for loop indices of
! OpenMP loop operations.
! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: define void @wsloop_i32
subroutine wsloop_i32()
  ! CHECK-DAG:  %[[LASTITER:.*]] = alloca i32
  ! CHECK-DAG:  %[[LB:.*]] = alloca i32
  ! CHECK-DAG:  %[[UB:.*]] = alloca i32
  ! CHECK-DAG:  %[[STRIDE:.*]] = alloca i32
  ! CHECK-DAG:  %[[I:.*]] = alloca i32
  integer :: i

  ! CHECK:      call void @llvm.lifetime.start.p0(i64 4, ptr %[[I]])
  ! CHECK-NEXT: br label %[[WSLOOP_BLOCK:.*]]
  ! CHECK:      [[WSLOOP_BLOCK]]:
  ! CHECK-NOT:  {{^.*}}:
  ! CHECK:      br label %[[CONT_BLOCK:.*]]
  ! CHECK:      [[CONT_BLOCK]]:
  ! CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 4, ptr %[[I]])
  !$omp do
  do i = 1, 10
    print *, i
  end do
  !$omp end do
end subroutine

! CHECK-LABEL: define void @wsloop_i64
subroutine wsloop_i64()
  ! CHECK-DAG:  %[[LASTITER:.*]] = alloca i32
  ! CHECK-DAG:  %[[LB:.*]] = alloca i64
  ! CHECK-DAG:  %[[UB:.*]] = alloca i64
  ! CHECK-DAG:  %[[STRIDE:.*]] = alloca i64
  ! CHECK-DAG:  %[[I:.*]] = alloca i64
  integer*8 :: i

  ! CHECK:      call void @llvm.lifetime.start.p0(i64 8, ptr %[[I]])
  ! CHECK-NEXT: br label %[[WSLOOP_BLOCK:.*]]
  ! CHECK:      [[WSLOOP_BLOCK]]:
  ! CHECK-NOT:  {{^.*}}:
  ! CHECK:      br label %[[CONT_BLOCK:.*]]
  ! CHECK:      [[CONT_BLOCK]]:
  ! CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr %[[I]])
  !$omp do
  do i = 1, 10
    print *, i
  end do
  !$omp end do
end subroutine

! CHECK-LABEL: define void @simdloop_i32
subroutine simdloop_i32()
  ! CHECK:      %[[I:.*]] = alloca i32
  integer :: i

  ! CHECK:      call void @llvm.lifetime.start.p0(i64 4, ptr %[[I]])
  ! CHECK-NEXT: br label %[[SIMDLOOP_BLOCK:.*]]
  ! CHECK:      [[SIMDLOOP_BLOCK]]:
  ! CHECK-NOT:  {{^.*}}:
  ! CHECK:      br label %[[CONT_BLOCK:.*]]
  ! CHECK:      [[CONT_BLOCK]]:
  ! CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 4, ptr %[[I]])
  !$omp simd
  do i=1, 9
    print *, i
  end do
  !$omp end simd 
end subroutine

! CHECK-LABEL: define void @simdloop_i64
subroutine simdloop_i64()
  ! CHECK:      %[[I:.*]] = alloca i64
  integer*8 :: i

  ! CHECK:      call void @llvm.lifetime.start.p0(i64 8, ptr %[[I]])
  ! CHECK-NEXT: br label %[[SIMDLOOP_BLOCK:.*]]
  ! CHECK:      [[SIMDLOOP_BLOCK]]:
  ! CHECK-NOT:  {{^.*}}:
  ! CHECK:      br label %[[CONT_BLOCK:.*]]
  ! CHECK:      [[CONT_BLOCK]]:
  ! CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr %[[I]])
  !$omp simd
  do i=1, 9
    print *, i
  end do
  !$omp end simd 
end subroutine
