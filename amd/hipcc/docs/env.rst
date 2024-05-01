.. meta::
  :description: HIPCC environment variables
  :keywords: HIPCC, ROCm, HIP tools, HIP compiler

.. _hipcc_vars:

******************************************
HIPCC environment variables
******************************************

The environment variable ``HIP_PLATFORM`` can be used to specify ``amd`` or ``nvidia`` depending on the available backend tool flows:

* ``HIP_PLATFORM``='amd' or ``HIP_PLATFORM``='nvidia'.

.. note:: 
    If ``HIP_PLATFORM`` is not set, then ``hipcc`` will attempt to auto-detect based on if the ``nvcc`` tool is found.

Additional environment variable controls:

* ``HIP_PATH``        : Path to HIP directory. The default is one dir level above the location of ``hipcc``.
* ``CUDA_PATH``       : Path to the CUDA SDK. The default is ``/usr/local/cuda``. This is only used for NVIDIA platforms.
* ``HIP_ROCCLR_HOME`` : Path to ``HIP/ROCclr`` directory. This is only used for AMD platforms.
* ``HIP_CLANG_PATH``  : Path to HIP-Clang. The default is ``../../llvm/bin`` relative to the absolute path of ``hipcc``). This is only used for AMD platforms.
