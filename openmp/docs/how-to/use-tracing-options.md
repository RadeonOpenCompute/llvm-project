

### Using tracing options

#### Prerequisite

When using the `--sys-trace` option, compile the OpenMP program with:

```bash

    -Wl,-rpath,/opt/rocm-{version}/lib -lamdhip64

```

The following tracing options are widely used to generate useful information:

* **`--hsa-trace`**: This option is used to get a JSON output file with the HSA API execution traces and a flat profile in a CSV file.

* **`--sys-trace`**: This allows programmers to trace both HIP and HSA calls. Since this option results in loading ``libamdhip64.so``, follow the
  prerequisite as mentioned above.

A CSV and a JSON file are produced by the above trace options. The CSV file presents the data in a tabular format, and the JSON file can be visualized using
Google Chrome at chrome://tracing/ or [Perfetto](https://perfetto.dev/). Navigate to Chrome or Perfetto and load the JSON file to see the timeline of the
HSA calls.

For more details on tracing, refer to the {doc}`ROCProfilerV1 User Manual <rocprofiler:rocprofv1>`.

### Environment variables

| Environment Variable        | Purpose                  |
| --------------------------- | ---------------------------- |
| `OMP_NUM_TEAMS`             | To set the number of teams for kernel launch, which is otherwise chosen by the implementation by default. You can set this number (subject to implementation limits) for performance tuning. |
| `LIBOMPTARGET_KERNEL_TRACE` | To print useful statistics for device operations. Setting it to 1 and running the program emits the name of every kernel launched, the number of teams and threads used, and the corresponding register usage. Setting it to 2 additionally emits timing information for kernel launches and data transfer operations between the host and the device. |
| `LIBOMPTARGET_INFO`         | To print informational messages from the device runtime as the program executes. Setting it to a value of 1 or higher, prints fine-grain information and setting it to -1 prints complete information. |
| `LIBOMPTARGET_DEBUG`        | To get detailed debugging information about data transfer operations and kernel launch when using a debug version of the device library. Set this environment variable to 1 to get the detailed information from the library. |
| `GPU_MAX_HW_QUEUES`         | To set the number of HSA queues in the OpenMP runtime. The HSA queues are created on demand up to the maximum value as supplied here. The queue creation starts with a single initialized queue to avoid unnecessary allocation of resources. The provided value is capped if it exceeds the recommended, device-specific value. |
| `LIBOMPTARGET_AMDGPU_MAX_ASYNC_COPY_BYTES` | To set the threshold size up to which data transfers are initiated asynchronously. The default threshold size is 1*1024*1024 bytes (1MB). |
| `OMPX_FORCE_SYNC_REGIONS` | To force the runtime to execute all operations synchronously, i.e., wait for an operation to complete immediately. This affects data transfers and kernel execution. While it is mainly designed for debugging, it may have a minor positive effect on performance in certain situations. |
:::
