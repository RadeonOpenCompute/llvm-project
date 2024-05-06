.. meta::
  :description: OpenMP
  :keywords: install, openmp, llvm, aomp, AMD, ROCm


The ROCm™ installation includes an LLVM-based implementation that fully supports the OpenMP 4.5 standard and a subset of OpenMP 5.0, 5.1, and 5.2 standards. Fortran, C/C++ compilers, and corresponding runtime libraries are included.
Along with host APIs, the OpenMP compilers support offloading code and data onto GPU devices. This document briefly describes the installation location of the OpenMP toolchain, example usage of device offloading, and usage of `rocprof` with OpenMP applications. The GPUs supported are the same as those supported by this ROCm release. See the list of supported GPUs for {doc}`Linux<rocm-install-on-linux:reference/system-requirements>` and {doc}`Windows<rocm-install-on-windows:reference/system-requirements>`.

The ROCm OpenMP compiler is implemented using LLVM compiler technology. The following image illustrates the internal steps taken to translate a user’s application into an executable that can offload computation to the AMDGPU. The compilation is a two-pass process. Pass 1 compiles the application to generate the CPU code and Pass 2 links the CPU code to the AMDGPU device code.

You can access  code on the `GitHub repository <https://github.com/ROCm/llvm-project>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`OpenMP installation <./install/install>`

  .. grid-item-card:: Conceptual

     * :doc:`OpenMP features <./conceptual/openmp-features>`

  .. grid-item-card:: How to

    * :doc:`<how-to/use-openmp>`
    * :doc:`<how-to/use-rocprof>`
    * :doc:`<how-to/use-tracing-options>`

  .. grid-item-card:: Reference

    * `OpenMP API specification for parallel programming <https://www.openmp.org/specifications/>`_    
    * :doc:`Command line argument reference <./reference/CommandLineArgumentReference>`
    * :doc:`OpenMP FAQ <./reference/faq>`
    * :doc:`Command line argument reference <./reference/CommandLineArgumentReference>`

  .. grid-item-card:: Tutorials

    * `GitHub samples <https://github.com/ROCm/rocDecode/tree/develop/samples>`_

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
