// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.alloca_memory_space", 5 : ui32>>, llvm.data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
  llvm.func @_QQmain() attributes {fir.bindc_name = "main", omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>} {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x i32 {bindc_name = "a"} : (i64) -> !llvm.ptr<5>
    %3 = llvm.addrspacecast %2 : !llvm.ptr<5> to !llvm.ptr
    omp.task {
      llvm.store %0, %3 : i32, !llvm.ptr
      omp.terminator
    }
    %4 = omp.map_info var_ptr(%3 : !llvm.ptr, i32) map_clauses(tofrom) capture(ByRef) -> !llvm.ptr {name = "a"}
    omp.target map_entries(%4 -> %arg0 : !llvm.ptr) {
    ^bb0(%arg0: !llvm.ptr):
      %5 = llvm.mlir.constant(5 : i32) : i32
      %6 = llvm.load %arg0  : !llvm.ptr -> i32
      %7 = llvm.add %6, %5  : i32
      llvm.store %7, %arg0  : i32, !llvm.ptr
      omp.terminator
    }
    llvm.return
  }
}

// This tests the fix for https://github.com/llvm/llvm-project/issues/84606
// We are only interested in ensuring that the -mlir-to-llmvir pass doesn't crash.
// CHECK: {{.*}} = add i32 {{.*}}, 5
