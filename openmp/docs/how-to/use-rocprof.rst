.. meta::
  :description: Install OpenMP
  :keywords: install, openmp, llvm, aomp, AMD, ROCm


Using `rocprof` with OpenMP
-----------------------------

The following steps describe a typical workflow for using `rocprof` with OpenMP code compiled with AOMP:

1. Run `rocprof` with the program command line:

  .. code-block:: bash
  
      % rocprof <application> <args>
    

    This produces a `results.csv` file in the user’s current directory that shows basic stats such as kernel names, grid size, number of registers used etc. The user can choose to specify the preferred output file name using the
    o option.

2. Add options for a detailed result:

   .. code-block:: bash
   
      --stats: % rocprof --stats <application> <args>
   

   The stats option produces timestamps for the kernels. Look into the output CSV file for the field, `DurationNs`, which is useful in getting an understanding of the critical kernels in the code.

   Apart from `--stats`, the option `--timestamp` on produces a timestamp for the kernels.

3. After learning about the required kernels, the user can take a detailed look at each one of them. `rocprof` has support for hardware counters: a set of basic and a set of derived ones. See the complete list of counters using
   options --list-basic and --list-derived. `rocprof` accepts either a text or an XML file as an input.

For more details on `rocprof`, refer to the {doc}`ROCProfilerV1 User Manual <rocprofiler:rocprofv1>`.
