.. meta::
  :description: Install OpenMP
  :keywords: install, openmp, llvm, aomp, AMD, ROCm


Using OpenMP
---------------

The example programs can be compiled and run by pointing the environment variable `ROCM_PATH` to the ROCm install directory.

Example
========

.. code-block:: bash

    export ROCM_PATH=/opt/rocm-{version}
    cd $ROCM_PATH/share/openmp-extras/examples/openmp/veccopy
    sudo make run


.. note::

`sudo` is required since we are building inside the `/opt` directory. Alternatively, copy the files to your home directory first.


The above invocation of Make compiles and runs the program. Note the options that are required for target offload from an OpenMP program:

.. code-block:: bash

    -fopenmp --offload-arch=<gpu-arch>


.. note:: 

The compiler also accepts the alternative offloading notation:

.. code-block:: bash

    -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=<gpu-arch>


Obtain the value of `gpu-arch` by running the following command:

.. code-block:: bash

    % /opt/rocm-{version}/bin/rocminfo | grep gfx


[//]: # (dated link below, needs updating)

See the complete list of compiler command-line references `here <https://github.com/ROCm/llvm-project/blob/amd-stg-open/clang/docs/CommandGuide/clang.rst>`_.

