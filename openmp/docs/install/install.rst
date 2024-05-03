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
