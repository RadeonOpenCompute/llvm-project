// RUN: mlir-translate -mlir-to-llvmir %s 2>&1 | FileCheck %s

// Set a dummy target triple to enable target region outlining.
module attributes {omp.target_triples = ["amdgcn-amd-amdhsa"]} {
  llvm.func @foo(%0 : i1) {
    omp.target if(%0) {
      omp.terminator
    }
    llvm.return
  }

// CHECK: define void @foo(i1 %[[COND:.*]]) {

// CHECK: br i1 %[[COND]], label %[[THEN_LABEL:.*]], label %[[ELSE_LABEL:.*]]

// CHECK: [[THEN_LABEL]]:
// CHECK: %[[RES:.*]] = call i32 @__tgt_target_kernel({{.*}})
// CHECK-NEXT: %[[OFFLOAD_CHECK:.*]] = icmp ne i32 %[[RES]], 0
// CHECK-NEXT: br i1 %[[OFFLOAD_CHECK]], label %[[OFF_FAIL_LABEL:.*]], label %[[OFF_SUCC_LABEL:.*]]

// CHECK: [[OFF_FAIL_LABEL]]:
// CHECK-NEXT: call void @[[FALLBACK_FN:.*]]()
// CHECK-NEXT: br label %[[OFF_CONT_LABEL:.*]]

// CHECK: [[OFF_CONT_LABEL]]:
// CHECK-NEXT: br label %[[END_LABEL:.*]]

// CHECK: [[ELSE_LABEL]]:
// CHECK-NEXT: call void @[[FALLBACK_FN]]()
// CHECK-NEXT: br label %[[END_LABEL]]

// CHECK: [[END_LABEL]]:
// CHECK-NEXT: ret void
}
