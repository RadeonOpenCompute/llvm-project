.. meta::
  :description: Install OpenMP
  :keywords: install, openmp, llvm, aomp, AMD, ROCm


**************
Installation
**************
The OpenMP toolchain is automatically installed as part of the standard ROCm installation and is available under /opt/rocm-{version}/llvm. The sub-directories are:

* bin: Compilers (flang and clang) and other binaries.
* examples: The usage section below shows how to compile and run these programs.
* include: Header files.
* lib: Libraries including those required for target offload.
* lib-debug: Debug versions of the above libraries.

Prerequisites
----------------

* Linux Kernel versions above 5.14
* Latest KFD driver packaged in ROCm stack
* Xnack, as USM support can only be tested with applications compiled with Xnack capability

Xnack capability
=================

When enabled, Xnack capability allows GPU threads to access CPU (system) memory, allocated with OS-allocators, such as malloc, new, and mmap. Xnack must be enabled both at compile- and run-time. To enable Xnack support at compile-time, use:

`--offload-arch=gfx908:xnack+`
Or use another functionally equivalent option Xnack-any:

`--offload-arch=gfx908`
To enable Xnack functionality at runtime on a per-application basis, use environment variable:

`HSA_XNACK=1`
When Xnack support is not needed:

Building OpenMP
================

Build the applications to maximize resource utilization using:
`--offload-arch=gfx908:xnack-`
At runtime, set the HSA_XNACK environment variable to 0.

Unified shared memory pragma
==============================

This OpenMP pragma is available on MI200 through xnack+ support.

omp requires unified_shared_memory
====================================
As stated in the OpenMP specifications, this pragma makes the map clause on target constructs optional. By default, on MI200, all memory allocated on the host is fine grain. Using the map clause on a target clause is allowed, which transforms the access semantics of the associated memory to coarse grain.

A simple program demonstrating the use of this feature is:

$ cat parallel_for.cpp
#include <stdlib.h>
#include <stdio.h>

#define N 64
#pragma omp requires unified_shared_memory

.. code-block:: bash

    int main() {
      int n = N;
      int *a = new int[n];
      int *b = new int[n];
    
      for(int i = 0; i < n; i++)
        b[i] = i;
    
      #pragma omp target parallel for map(to:b[:n])
      for(int i = 0; i < n; i++)
        a[i] = b[i];
    
      for(int i = 0; i < n; i++)
        if(a[i] != i)
          printf("error at %d: expected %d, got %d\n", i, i+1, a[i]);
    
      return 0;
    }
    $ clang++ -O2 -target x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx90a:xnack+ parallel_for.cpp
    $ HSA_XNACK=1 ./a.out

In the above code example, pointer “a” is not mapped in the target region, while pointer “b” is. Both are valid pointers on the GPU device and passed by-value to the kernel implementing the target region. This means the pointer values on the host and the device are the same.

The difference between the memory pages pointed to by these two variables is that the pages pointed by “a” are in fine-grain memory, while the pages pointed to by “b” are in coarse-grain memory during and after the execution of the target region. This is accomplished in the OpenMP runtime library with calls to the ROCr runtime to set the pages pointed by “b” as coarse grain.

