# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;RMM: RAPIDS Memory Manager</div>

[![Build Status](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/rmm/job/branches/job/rmm-branch-pipeline/badge/icon)](https://gpuci.gpuopenanalytics.com/job/rapidsai/job/gpuci/job/rmm/job/branches/job/rmm-branch-pipeline/)

RAPIDS Memory Manager (RMM) is:

 - A replacement allocator for CUDA Device Memory (and CUDA Managed Memory).
 - A pool allocator to make CUDA device memory allocation / deallocation faster
   and asynchronous.
 - A central place for all device memory allocations in cuDF (C++ and Python) and
   other [RAPIDS](https://rapids.ai) libraries.

RMM is not:

 - A replacement allocator for host memory (`malloc`, `new`, `cudaMallocHost`,
   `cudaHostRegister`).

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/rmm/blob/master/README.md) ensure you are on the `master` branch.

## Installation

### Conda

RMM can be installed with conda ([miniconda](https://conda.io/miniconda.html), or the full [Anaconda distribution](https://www.anaconda.com/download)) from the `rapidsai` channel:

For `rmm version == 0.12` :
```bash
# for CUDA 10.1
conda install -c nvidia -c rapidsai-nightly -c conda-forge -c defaults \
    rmm=0.12 python=3.6 cudatoolkit=10.1

# or, for CUDA 10.0
conda install -c nvidia -c rapidsai-nightly -c conda-forge -c defaults \
    rmm=0.12 python=3.6 cudatoolkit=10.0
```

For `rmm version == 0.11` :
```bash
# for CUDA 10.1
conda install -c nvidia -c rapidsai -c conda-forge -c defaults \
    rmm=0.11 python=3.6 cudatoolkit=10.1

# or, for CUDA 10.0
conda install -c nvidia -c rapidsai -c conda-forge -c defaults \
    rmm=0.11 python=3.6 cudatoolkit=10.0
```
We also provide [nightly conda packages](https://anaconda.org/rapidsai-nightly) built from the tip of our latest development branch.

Note: RMM is supported only on Linux, and with Python versions 3.6 or 3.7.

See the [Get RAPIDS version picker](https://rapids.ai/start.html) for more OS and version info.

## Building from Source

### Get RMM Dependencies

Compiler requirements:

* `gcc`     version 4.8 or higher recommended
* `nvcc`    version 9.0 or higher recommended
* `cmake`   version 3.12 or higher

CUDA/GPU requirements:

* CUDA 9.0+
* NVIDIA driver 396.44+
* Pascal architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

### Script to build RMM from source

To install RMM from source, ensure the dependencies are met and follow the steps below:

- Clone the repository and submodules
```bash
$ git clone --recurse-submodules https://github.com/rapidsai/rmm.git
$ cd rmm
```

Follow the instructions under "Create the conda development environment `cudf_dev`" in the [cuDF README](https://github.com/rapidsai/cudf#build-from-source).

- Create the conda development environment `cudf_dev`
```bash
# create the conda environment (assuming in base `cudf` directory)
$ conda env create --name cudf_dev --file conda/environments/dev_py35.yml
# activate the environment
$ source activate cudf_dev
```

- Build and install `librmm` using cmake & make. CMake depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.
```bash

$ mkdir build                                       # make a build directory
$ cd build                                          # enter the build directory
$ cmake .. -DCMAKE_INSTALL_PREFIX=/install/path     # configure cmake ... use $CONDA_PREFIX if you're using Anaconda
$ make -j                                           # compile the library librmm.so ... '-j' will start a parallel job using the number of physical cores available on your system
$ make install                                      # install the library librmm.so to '/install/path'
```

- Building and installing `librmm` and `rmm` using build.sh. Build.sh creates build dir at root of git repository. build.sh depends on the `nvcc` executable being on your path or defined in `$CUDACXX`.
```bash

$ ./build.sh -h                                     # Display help and exit
$ ./build.sh -n librmm                              # Build librmm without installing
$ ./build.sh -n rmm                                 # Build rmm without installing
$ ./build.sh -n librmm rmm                          # Build librmm and rmm without installing
$ ./build.sh librmm rmm                             # Build and install librmm and rmm
```

- To run tests (Optional):
```bash
$ cd build (if you are not already in build directory)
$ make test
```

- Build, install, and test the `rmm` python package, in the `python` folder:
```bash
$ python setup.py build_ext --inplace
$ python setup.py install
$ pytest -v
```

Done! You are ready to develop for the RMM OSS project.

## Using RMM in C/C++ code

Using RMM in CUDA C++ code is straightforward. Include `rmm.h` and replace calls
to `cudaMalloc()` and `cudaFree()` with calls to the `RMM_ALLOC()` and
`RMM_FREE()` macros, respectively.

Note that `RMM_ALLOC` and `RMM_FREE` take an additional parameter, a stream
identifier. This is necessary to enable asynchronous allocation and
deallocation; however, the default (also known as null) stream (or `0`) can be
used. For example:

```
// old
cudaError_t result = cudaMalloc(&myvar, size_in_bytes);
// ...
cudaError_t result = cudaFree(myvar);
```

```
// new
rmmError_t result = RMM_ALLOC(&myvar, size_in_bytes, stream_id);
// ...
rmmError_t result = RMM_FREE(myvar, stream_id);
```

Note that `RMM_ALLOC` and `RMM_FREE` are wrappers around `rmm::alloc()` and
`rmm::free()`, respectively. The lower-level functions also take a file name and
a line number for tracking the location of RMM allocations and deallocations.
The macro versions use the C preprocessor to automatically specify these params.

### Using RMM with Thrust

RAPIDS and other CUDA libraries make heavy use of Thrust. Thrust uses CUDA device memory in two
situations:

 1. As the backing store for `thrust::device_vector`, and
 2. As temporary storage inside some algorithms, such as `thrust::sort`.

RMM includes a custom Thrust allocator in the file `thrust_rmm_allocator.h`. This defines the template class `rmm_allocator`, and
a custom Thrust CUDA device execution policy called `rmm::exec_policy(stream)`.

#### Thrust Device Vectors

Instead of creating device vectors like this:

```
thrust::device_vector<size_type> permuted_indices(column_length);
```

You can tell Thrust to use `rmm_allocator` like this:

```
thrust::device_vector<size_type, rmm_allocator<T>> permuted_indices(column_length);
```

For convenience, you can use the alias `rmm::device_vector<T>` defined in
`thrust_rmm_allocator.h` that can be used as if it were a `thrust::device_vector<T>`.

#### Thrust Algorithms

To instruct Thrust to use RMM to allocate temporary storage, you can use the custom
Thrust CUDA device execution policy: `rmm::exec_policy(stream)`.
This instructs Thrust to use the `rmm_allocator` on the specified stream for temporary memory allocation.

`rmm::exec_policy(stream)` returns a `std::unique_ptr` to a Thrust execution policy that uses `rmm_allocator` for temporary allocations.
In order to specify that the Thrust algorithm be executed on a specific stream, the usage is:

```
thrust::sort(rmm::exec_policy(stream)->on(stream), ...);
```

The first `stream` argument is the `stream` to use for `rmm_allocator`.
The second `stream` argument is what should be used to execute the Thrust algorithm.
These two arguments must be identical.


## Using RMM in Python Code

cuDF and other Python libraries typically create arrays of CUDA device memory
by using Numba's `cuda.device_array` interfaces. Until Numba provides a plugin
interface for using an external memory manager, RMM provides an API compatible
with `cuda.device_array` constructors that cuDF (also cuDF C++ API pytests)
should use to ensure all CUDA device memory is allocated via the memory manager.
RMM provides:

   - `rmm.device_array()`
   - `rmm.device_array_like()`
   - `rmm.to_device()`
   - `rmm.auto_device()`

Which are compatible with their Numba `cuda.*` equivalents. They return a Numba
NDArray object whose memory is allocated in CUDA device memory using RMM.

Following is an example from cuDF `groupby.py` that copies from a numpy array to
an equivalent CUDA `device_array` using `to_device()`, and creates a device
array using `device_array`, and then runs a Numba kernel (`group_mean`) to
compute the output values.

```
    ...
    dev_begins = rmm.to_device(np.asarray(begin))
    dev_out = rmm.device_array(size, dtype=np.float64)
    if size > 0:
        group_mean.forall(size)(sr.to_gpu_array(),
                                dev_begins,
                                dev_out)
    values[newk] = dev_out
```
In another example from cuDF `cudautils.py`, `fillna` uses `device_array_like`
to construct a CUDA device array with the same shape and data type as another.

```
def fillna(data, mask, value):
    out = rmm.device_array_like(data)
    out.copy_to_device(data)
    configured = gpu_fill_masked.forall(data.size)
    configured(value, mask, out)
    return out
```

`rmm` also provides `get_ipc_handle()` for getting the IPC handle associated
with a Numba NDArray, which accounts for the case where the data for the NDArray
is suballocated from some larger pool allocation by the memory manager.

### Handling RMM Options in Python Code

RMM currently defaults to just calling cudaMalloc, but you can enable the
experimental pool allocator by reinitializing RMM.

```
rmm.reinitialize(
    pool_allocator=False, # default is False
    managed_memory=False, # default is False
    initial_pool_size=int(2**31), # set to 2GiB. Default is 1/2 total GPU memory
    devices=0, # GPU device  IDs to register. By default registers only GPU 0.
    logging=True, # default is False -- has perf overhead
)
```

To configure RMM options to be used in cuDF before loading, simply do the above
before you `import cudf`. You can re-initialize the memory manager with
different settings at run time by calling `rmm.reinitialize()` with the above
options.

You can also optionally use the internal functions in cuDF which call these
functions. Here are some example configuration functions that can be used in
a notebook to initialize the memory manager in each Dask worker.

```python
import cudf


# Default passthrough to cudaMalloc
cudf.set_allocator()

# Use the pool allocator
cudf.set_allocator(pool=True)

# Use the pool allocator with a 2GiB initial pool size
cudf.set_allocator(pool=True, initial_pool_size=2<<30)
```

Remember that while the pool is in use memory is not freed. So if you follow
cuDF operations with device-memory-intensive computations that don't use RMM
(such as XGBoost), you will need to move the data to the host and then
finalize RMM. The Mortgage E2E workflow notebook uses this technique. We are
working on better ways to reclaim memory, as well as making RAPIDS machine
learning libraries use the same RMM memory pool.

### Memory info

The amount of free and total memory managed by RMM associated with a particular
stream can be obtained with the `get_info` function:

```python
meminfo = rmm.get_info()
print(meminfo.free)  # E.g. "16046292992"
print(meminfo.total) # E.g. "16914055168"
```

### CUDA Managed Memory

RMM can be set to allocate all memory as managed memory (`cudaMallocManaged`
underlying allocator). This is enabled in C++ by setting the `allocation_mode`
member of the struct `rmmOptions_t` to include the flag `CudaManagedMemory`
(the flags are ORed), and passing it to `rmmInitialize()`. If the flag
`PoolAllocation` is also set, then RMM will allocate from a pool of managed
memory.

When the allocation mode is both `CudaManagedMemory` and `PoolAllocation`,
RMM allocates the initial pool (and any expansion allocations) using
`cudaMallocManaged` and then prefetches the pool to the GPU using
`cudaMemPrefetchAsync` so all pool memory that will fit is initially located
on the device.

### Thread Safety

RMM aims to be thread safe, and provides the following guarantees.

1. `rmmInitialize()` and `rmmFinalize()` are thread-safe with respect to each other. In other words, `rmmFinalize()` cannot interrupt `rmmInitialize()` and vice versa.
2. `rmmAlloc()` and `rmmFree()` calls are thread-safe with respect to each other. In other words, the state of the underlying allocator is thread safe.
3. `rmmAlloc()` and `rmmFree()` are NOT thread-safe with respect to `rmmInitialize()` and `rmmFinalize()`. For example, a `rmmFinalize()` *could* interrupt an `rmmAlloc()` or `rmmFree()` in another thread. Therefore applications should ensure to protect calls to `rmmInitialize()` `rmmFinalize()`.
