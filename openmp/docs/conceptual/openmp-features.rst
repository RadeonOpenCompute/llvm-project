.. meta::
  :description: Install OpenMP
  :keywords: install, openmp, llvm, aomp, AMD, ROCm

*****************
OpenMP features
*****************
  
The OpenMP programming model is greatly enhanced with the following new features implemented in the past releases.

(openmp_usm)=

Asynchronous behavior in OpenMP target regions
----------------------------------------------
  
* Controlling Asynchronous Behavior - The OpenMP offloading runtime executes in an asynchronous fashion by default, allowing multiple data transfers to start concurrently. 
  However, if the data to be transferred becomes larger than the default threshold of 1MB, the runtime falls back to a synchronous data transfer. The buffers that have been locked already are always executed asynchronously. You can overrule this default behavior by setting LIBOMPTARGET_AMDGPU_MAX_ASYNC_COPY_BYTES and OMPX_FORCE_SYNC_REGIONS. See the Environment Variables table for details.

* Multithreaded Offloading on the Same Device - The libomptarget plugin for GPU offloading allows creation of separate configurable HSA queues per chiplet, which enables two or more threads to concurrently offload to the same device.

* Parallel Memory Copy Invocations - Implicit asynchronous execution of single target region enables parallel memory copy invocations.

* Unified shared memory - Unified Shared Memory (USM) provides a pointer-based approach to memory management. To implement USM, fulfill the following system requirements along with Xnack capability.


OMPT target support
---------------------

The OpenMP runtime in ROCm implements a subset of the OMPT device APIs, as described in the OpenMP specification document. These APIs allow first-party tools to examine the profile and kernel traces that execute on a device. A tool can register callbacks for data transfer and kernel dispatch entry points or use APIs to start and stop tracing for device-related activities such as data transfer and kernel dispatch timings and associated metadata. If device tracing is enabled, trace records for device activities are collected during program execution and returned to the tool using the APIs described in the specification.

The following example demonstrates how a tool uses the supported OMPT target APIs. The README in /opt/rocm/llvm/examples/tools/ompt outlines the steps to be followed, and the provided example can be run as follows:

.. code-block::

    cd $ROCM_PATH/share/openmp-extras/examples/tools/ompt/veccopy-ompt-target-tracing
    sudo make run

The file veccopy-ompt-target-tracing.c simulates how a tool initiates device activity tracing. The file callbacks.h shows the callbacks registered and implemented by the tool.

Floating point atomic operations
The MI200-series GPUs support the generation of hardware floating-point atomics using the OpenMP atomic pragma. The support includes single- and double-precision floating-point atomic operations. The programmer must ensure that the memory subjected to the atomic operation is in coarse-grain memory by mapping it explicitly with the help of map clauses when not implicitly mapped by the compiler as per the OpenMP specifications. This makes these hardware floating-point atomic instructions “fast,” as they are faster than using a default compare-and-swap loop scheme, but at the same time “unsafe,” as they are not supported on fine-grain memory. The operation in unified_shared_memory mode also requires programmers to map the memory explicitly when not implicitly mapped by the compiler.

To request fast floating-point atomic instructions at the file level, use compiler flag -munsafe-fp-atomics or a hint clause on a specific pragma:

double a = 0.0;
#pragma omp atomic hint(AMD_fast_fp_atomics)
a = a + 1.0;
:::{note} AMD_unsafe_fp_atomics is an alias for AMD_fast_fp_atomics, and AMD_safe_fp_atomics is implemented with a compare-and-swap loop. :::

To disable the generation of fast floating-point atomic instructions at the file level, build using the option -msafe-fp-atomics or use a hint clause on a specific pragma:

double a = 0.0;
#pragma omp atomic hint(AMD_safe_fp_atomics)
a = a + 1.0;
The hint clause value always has a precedence over the compiler flag, which allows programmers to create atomic constructs with a different behavior than the rest of the file.

See the example below, where the user builds the program using -msafe-fp-atomics to select a file-wide “safe atomic” compilation. However, the fast atomics hint clause over variable “a” takes precedence and operates on “a” using a fast/unsafe floating-point atomic, while the variable “b” in the absence of a hint clause is operated upon using safe floating-point atomics as per the compiler flag.

double a = 0.0;.
#pragma omp atomic hint(AMD_fast_fp_atomics)
a = a + 1.0;

double b = 0.0;
#pragma omp atomic
b = b + 1.0;
AddressSanitizer tool
AddressSanitizer (ASan) is a memory error detector tool utilized by applications to detect various errors ranging from spatial issues such as out-of-bound access to temporal issues such as use-after-free. The AOMP compiler supports ASan for AMD GPUs with applications written in both HIP and OpenMP.

Features supported on host platform (Target x86_64):

Use-after-free
Buffer overflows
Heap buffer overflow
Stack buffer overflow
Global buffer overflow
Use-after-return
Use-after-scope
Initialization order bugs
Features supported on AMDGPU platform (amdgcn-amd-amdhsa):

Heap buffer overflow
Global buffer overflow
Software (kernel/OS) requirements: Unified Shared Memory support with Xnack capability. See the section on Unified Shared Memory for prerequisites and details on Xnack.

Example:

Heap buffer overflow
void  main() {
.......  // Some program statements
.......  // Some program statements
#pragma omp target map(to : A[0:N], B[0:N]) map(from: C[0:N])
{
#pragma omp parallel for
    for(int i =0 ; i < N; i++){
    C[i+10] = A[i] + B[i];
  }   // end of for loop
}
.......   // Some program statements
}// end of main
See the complete sample code for heap buffer overflow here.

Global buffer overflow
#pragma omp declare target
   int A[N],B[N],C[N];
#pragma omp end declare target
void main(){
......  // some program statements
......  // some program statements
#pragma omp target data map(to:A[0:N],B[0:N]) map(from: C[0:N])
{
#pragma omp target update to(A,B)
#pragma omp target parallel for
for(int i=0; i<N; i++){
    C[i]=A[i*100]+B[i+22];
} // end of for loop
#pragma omp target update from(C)
}
........  // some program statements
} // end of main
See the complete sample code for global buffer overflow here.

Clang compiler option for kernel optimization
You can use the clang compiler option -fopenmp-target-fast for kernel optimization if certain constraints implied by its component options are satisfied. -fopenmp-target-fast enables the following options:

-fopenmp-target-ignore-env-vars: It enables code generation of specialized kernels including no-loop and Cross-team reductions.

-fopenmp-assume-no-thread-state: It enables the compiler to assume that no thread in a parallel region modifies an Internal Control Variable (ICV), thus potentially reducing the device runtime code execution.

-fopenmp-assume-no-nested-parallelism: It enables the compiler to assume that no thread in a parallel region encounters a parallel region, thus potentially reducing the device runtime code execution.

-O3 if no -O* is specified by the user.

Specialized kernels
Clang will attempt to generate specialized kernels based on compiler options and OpenMP constructs. The following specialized kernels are supported:

No-loop
Big-jump-loop
Cross-team reductions
To enable the generation of specialized kernels, follow these guidelines:

Do not specify teams, threads, and schedule-related environment variables. The num_teams clause in an OpenMP target construct acts as an override and prevents the generation of the no-loop kernel. If the specification of num_teams clause is a user requirement then clang tries to generate the big-jump-loop kernel instead of the no-loop kernel.

Assert the absence of the teams, threads, and schedule-related environment variables by adding the command-line option -fopenmp-target-ignore-env-vars.

To automatically enable the specialized kernel generation, use -Ofast or -fopenmp-target-fast for compilation.

To disable specialized kernel generation, use -fno-openmp-target-ignore-env-vars.

No-loop kernel generation
The no-loop kernel generation feature optimizes the compiler performance by generating a specialized kernel for certain OpenMP target constructs such as target teams distribute parallel for. The specialized kernel generation feature assumes every thread executes a single iteration of the user loop, which leads the runtime to launch a total number of GPU threads equal to or greater than the iteration space size of the target region loop. This allows the compiler to generate code for the loop body without an enclosing loop, resulting in reduced control-flow complexity and potentially better performance.

Big-jump-loop kernel generation
A no-loop kernel is not generated if the OpenMP teams construct uses a num_teams clause. Instead, the compiler attempts to generate a different specialized kernel called the big-jump-loop kernel. The compiler launches the kernel with a grid size determined by the number of teams specified by the OpenMP num_teams clause and the blocksize chosen either by the compiler or specified by the corresponding OpenMP clause.

Cross-team optimized reduction kernel generation
If the OpenMP construct has a reduction clause, the compiler attempts to generate optimized code by utilizing efficient cross-team communication. New APIs for cross-team reduction are implemented in the device runtime and are automatically generated by clang.
