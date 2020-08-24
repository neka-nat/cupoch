# RMM 0.16.0 (Date TBD)

## New Features

## Improvements

- PR #477 Just use `None` for `strides` in `DeviceBuffer`

## Bug Fixes

# RMM 0.15.0 (Date TBD)

## New Features

- PR #375 Support out-of-band buffers in Python pickling
- PR #391 Add `get_default_resource_type`
- PR #396 Remove deprecated RMM APIs
- PR #425 Add CUDA per-thread default stream support and thread safety to `pool_memory_resource`
- PR #436 Always build and test with per-thread default stream enabled in the GPU CI build
- PR #444 Add `owning_wrapper` to simplify lifetime management of resources and their upstreams
- PR #449 Stream-ordered suballocator base class and per-thread default stream support 
          and thread safety for `fixed_size_memory_resource`
- PR #450 Add support for new build process (Project Flash)
- PR #457 New `binning_memory_resource` (replaces `hybrid_memory_resource` and 
          `fixed_multisize_memory_resource`).
- PR #458 Add `get/set_per_device_resource` to better support multi-GPU per process applications
- PR #466 Deprecate CNMeM.
- PR #489 Move `cudf._cuda` into `rmm._cuda`
- PR #504 Generate `gpu.pxd` based on cuda version as a preprocessor step
- PR #506 Upload rmm package per version python-cuda combo

## Improvements

- PR #428 Add the option to automatically flush memory allocate/free logs
- PR #378 Use CMake `FetchContent` to obtain latest release of `cub` and `thrust`
- PR #377 A better way to fetch `spdlog`
- PR #372 Use CMake `FetchContent` to obtain `cnmem` instead of git submodule
- PR #382 Rely on NumPy arrays for out-of-band pickling
- PR #386 Add short commit to conda package name
- PR #401 Update `get_ipc_handle()` to use cuda driver API
- PR #404 Make all memory resources thread safe in Python
- PR #402 Install dependencies via rapids-build-env
- PR #405 Move doc customization scripts to Jenkins
- PR #427 Add DeviceBuffer.release() cdef method
- PR #414 Add element-wise access for device_uvector
- PR #421 Capture thread id in logging and improve logger testing
- PR #426 Added multi-threaded support to replay benchmark
- PR #429 Fix debug build and add new CUDA assert utility
- PR #435 Update conda upload versions for new supported CUDA/Python
- PR #437 Test with `pickle5` (for older Python versions)
- PR #443 Remove thread safe adaptor from PoolMemoryResource
- PR #445 Make all resource operators/ctors explicit
- PR #447 Update Python README with info about DeviceBuffer/MemoryResource and external libraries
- PR #456 Minor cleanup: always use rmm/-prefixed includes
- PR #461 cmake improvements to be more target-based
- PR #468 update past release dates in changelog
- PR #486 Document relationship between active CUDA devices and resources
- PR #493 Rely on C++ lazy Memory Resource initialization behavior instead of initializing in Python

## Bug Fixes

- PR #433 Fix python imports
- PR #400 Fix segfault in RANDOM_ALLOCATIONS_BENCH
- PR #383 Explicitly require NumPy
- PR #398 Fix missing head flag in merge_blocks (pool_memory_resource) and improve block class
- PR #403 Mark Cython `memory_resource_wrappers` `extern` as `nogil`
- PR #406 Sets Google Benchmark to a fixed version, v1.5.1.
- PR #434 Fix issue with incorrect docker image being used in local build script
- PR #463 Revert cmake change for cnmem header not being added to source directory
- PR #464 More completely revert cnmem.h cmake changes
- PR #473 Fix initialization logic in pool_memory_resource
- PR #479 Fix usage of block printing in pool_memory_resource
- PR #490 Allow importing RMM without initializing CUDA driver
- PR #484 Fix device_uvector copy constructor compilation error and add test
- PR #498 Max pool growth less greedy
- PR #500 Use tempfile rather than hardcoded path in `test_rmm_csv_log`
- PR #511 Specify `--basetemp` for `py.test` run
- PR #509 Fix missing : before __LINE__ in throw string of RMM_CUDA_TRY
- PR #510 Fix segfault in pool_memory_resource when a CUDA stream is destroyed


# RMM 0.14.0 (03 Jun 2020)

## New Features

- PR #317 Provide External Memory Management Plugin for Numba
- PR #362 Add spdlog as a dependency in the conda package
- PR #360 Support logging to stdout/stderr
- PR #341 Enable logging
- PR #343 Add in option to statically link against cudart
- PR #364 Added new uninitialized device vector type, `device_uvector`

## Improvements

- PR #369 Use CMake `FetchContent` to obtain `spdlog` instead of vendoring
- PR #366 Remove installation of extra test dependencies
- PR #354 Add CMake option for per-thread default stream
- PR #350 Add .clang-format file & format all files
- PR #358 Fix typo in `rmm_cupy_allocator` docstring
- PR #357 Add Docker 19 support to local gpuci build
- PR #365 Make .clang-format consistent with cuGRAPH and cuDF
- PR #371 Add docs build script to repository
- PR #363 Expose `memory_resources` in Python

## Bug Fixes

- PR #373 Fix build.sh
- PR #346 Add clearer exception message when RMM_LOG_FILE is unset
- PR #347 Mark rmmFinalizeWrapper nogil
- PR #348 Fix unintentional use of pool-managed resource.
- PR #367 Fix flake8 issues
- PR #368 Fix `clang-format` missing comma bug
- PR #370 Fix stream and mr use in `device_buffer` methods
- PR #379 Remove deprecated calls from synchronization.cpp
- PR #381 Remove test_benchmark.cpp from cmakelists
- PR #392 SPDLOG matches other header-only acquisition patterns


# RMM 0.13.0 (31 Mar 2020)

## New Features

- PR #253 Add `frombytes` to convert `bytes`-like to `DeviceBuffer`
- PR #252 Add `__sizeof__` method to `DeviceBuffer`
- PR #258 Define pickling behavior for `DeviceBuffer`
- PR #261 Add `__bytes__` method to `DeviceBuffer`
- PR #262 Moved device memory resource files to `mr/device` directory
- PR #266 Drop `rmm.auto_device`
- PR #268 Add Cython/Python `copy_to_host` and `to_device`
- PR #272 Add `host_memory_resource`.
- PR #273 Moved device memory resource tests to `device/` directory.
- PR #274 Add `copy_from_host` method to `DeviceBuffer`
- PR #275 Add `copy_from_device` method to `DeviceBuffer`
- PR #283 Add random allocation benchmark.
- PR #287 Enabled CUDA CXX11 for unit tests.
- PR #292 Revamped RMM exceptions.
- PR #297 Use spdlog to implement `logging_resource_adaptor`.
- PR #303 Added replay benchmark.
- PR #319 Add `thread_safe_resource_adaptor` class.
- PR #314 New suballocator memory_resources.
- PR #330 Fixed incorrect name of `stream_free_blocks_` debug symbol.
- PR #331 Move to C++14 and deprecate legacy APIs.

## Improvements

- PR #246 Type `DeviceBuffer` arguments to `__cinit__`
- PR #249 Use `DeviceBuffer` in `device_array`
- PR #255 Add standard header to all Cython files
- PR #256 Cast through `uintptr_t` to `cudaStream_t`
- PR #254 Use `const void*` in `DeviceBuffer.__cinit__`
- PR #257 Mark Cython-exposed C++ functions that raise
- PR #269 Doc sync behavior in `copy_ptr_to_host`
- PR #278 Allocate a `bytes` object to fill up with RMM log data
- PR #280 Drop allocation/deallocation of `offset`
- PR #282 `DeviceBuffer` use default constructor for size=0
- PR #296 Use CuPy's `UnownedMemory` for RMM-backed allocations
- PR #310 Improve `device_buffer` allocation logic.
- PR #309 Sync default stream in `DeviceBuffer` constructor
- PR #326 Sync only on copy construction
- PR #308 Fix typo in README
- PR #334 Replace `rmm_allocator` for Thrust allocations
- PR #345 Remove stream synchronization from `device_scalar` constructor and `set_value`

## Bug Fixes

- PR #298 Remove RMM_CUDA_TRY from cuda_event_timer destructor
- PR #299 Fix assert condition blocking debug builds
- PR #300 Fix host mr_tests compile error
- PR #312 Fix libcudf compilation errors due to explicit defaulted device_buffer constructor


# RMM 0.12.0 (04 Feb 2020)

## New Features

- PR #218 Add `_DevicePointer`
- PR #219 Add method to copy `device_buffer` back to host memory
- PR #222 Expose free and total memory in Python interface
- PR #235 Allow construction of `DeviceBuffer` with a `stream`

## Improvements

- PR #214 Add codeowners
- PR #226 Add some tests of the Python `DeviceBuffer`
- PR #233 Reuse the same `CUDA_HOME` logic from cuDF
- PR #234 Add missing `size_t` in `DeviceBuffer`
- PR #239 Cleanup `DeviceBuffer`'s `__cinit__`
- PR #242 Special case 0-size `DeviceBuffer` in `tobytes`
- PR #244 Explicitly force `DeviceBuffer.size` to an `int`
- PR #247 Simplify casting in `tobytes` and other cleanup

## Bug Fixes

- PR #215 Catch polymorphic exceptions by reference instead of by value
- PR #221 Fix segfault calling rmmGetInfo when uninitialized
- PR #225 Avoid invoking Python operations in c_free
- PR #230 Fix duplicate symbol issues with `copy_to_host`
- PR #232 Move `copy_to_host` doc back to header file


# RMM 0.11.0 (11 Dec 2019)

## New Features

- PR #106 Added multi-GPU initialization
- PR #167 Added value setter to `device_scalar`
- PR #163 Add Cython bindings to `device_buffer`
- PR #177 Add `__cuda_array_interface__` to `DeviceBuffer`
- PR #198 Add `rmm.rmm_cupy_allocator()`

## Improvements

- PR #161 Use `std::atexit` to finalize RMM after Python interpreter shutdown
- PR #165 Align memory resource allocation sizes to 8-byte
- PR #171 Change public API of RMM to only expose `reinitialize(...)`
- PR #175 Drop `cython` from run requirements
- PR #169 Explicit stream argument for device_buffer methods
- PR #186 Add nbytes and len to DeviceBuffer
- PR #188 Require kwargs in `DeviceBuffer`'s constructor
- PR #194 Drop unused imports from `device_buffer.pyx`
- PR #196 Remove unused CUDA conda labels
- PR #200 Simplify DeviceBuffer methods

## Bug Fixes

- PR #174 Make `device_buffer` default ctor explicit to work around type_dispatcher issue in libcudf.
- PR #170 Always build librmm and rmm, but conditionally upload based on CUDA / Python version
- PR #182 Prefix `DeviceBuffer`'s C functions
- PR #189 Drop `__reduce__` from `DeviceBuffer`
- PR #193 Remove thrown exception from `rmm_allocator::deallocate`
- PR #224 Slice the CSV log before converting to bytes


# RMM 0.10.0 (16 Oct 2019)

## New Features

- PR #99 Added `device_buffer` class
- PR #133 Added `device_scalar` class

## Improvements

- PR #123 Remove driver install from ci scripts
- PR #131 Use YYMMDD tag in nightly build
- PR #137 Replace CFFI python bindings with Cython
- PR #127 Use Memory Resource classes for allocations

## Bug Fixes

- PR #107 Fix local build generated file ownerships
- PR #110 Fix Skip Test Functionality
- PR #125 Fixed order of private variables in LogIt
- PR #139 Expose `_make_finalizer` python API needed by cuDF
- PR #142 Fix ignored exceptions in Cython
- PR #146 Fix rmmFinalize() not freeing memory pools
- PR #149 Force finalization of RMM objects before RMM is finalized (Python)
- PR #154 Set ptr to 0 on rmm::alloc error
- PR #157 Check if initialized before freeing for Numba finalizer and use `weakref` instead of `atexit`


# RMM 0.9.0 (21 Aug 2019)

## New Features

- PR #96 Added `device_memory_resource` for beginning of overhaul of RMM design
- PR #103 Add and use unified build script

## Improvements

- PR #111 Streamline CUDA_REL environment variable
- PR #113 Handle ucp.BufferRegion objects in auto_device

## Bug Fixes

   ...


# RMM 0.8.0 (27 June 2019)

## New Features

- PR #95 Add skip test functionality to build.sh

## Improvements

   ...

## Bug Fixes

- PR #92 Update docs version


# RMM 0.7.0 (10 May 2019)

## New Features

- PR #67 Add random_allocate microbenchmark in tests/performance
- PR #70 Create conda environments and conda recipes
- PR #77 Add local build script to mimic gpuCI
- PR #80 Add build script for docs

## Improvements

- PR #76 Add cudatoolkit conda dependency
- PR #84 Use latest release version in update-version CI script
- PR #90 Avoid using c++14 auto return type for thrust_rmm_allocator.h

## Bug Fixes

- PR #68 Fix signed/unsigned mismatch in random_allocate benchmark
- PR #74 Fix rmm conda recipe librmm version pinning
- PR #72 Remove unnecessary _BSD_SOURCE define in random_allocate.cpp


# RMM 0.6.0 (18 Mar 2019)

## New Features

- PR #43 Add gpuCI build & test scripts
- PR #44 Added API to query whether RMM is initialized and with what options.
- PR #60 Default to CXX11_ABI=ON

## Improvements

## Bug Fixes

- PR #58 Eliminate unreliable check for change in available memory in test
- PR #49 Fix pep8 style errors detected by flake8


# RMM 0.5.0 (28 Jan 2019)

## New Features

- PR #2 Added CUDA Managed Memory allocation mode

## Improvements

- PR #12 Enable building RMM as a submodule
- PR #13 CMake: Added CXX11ABI option and removed Travis references
- PR #16 CMake: Added PARALLEL_LEVEL environment variable handling for GTest build parallelism (matches cuDF)
- PR #17 Update README with v0.5 changes including Managed Memory support

## Bug Fixes

- PR #10 Change cnmem submodule URL to use https
- PR #15 Temporarily disable hanging AllocateTB test for managed memory
- PR #28 Fix invalid reference to local stack variable in `rmm::exec_policy`


# RMM 0.4.0 (20 Dec 2018)

## New Features

- PR #1 Spun off RMM from cuDF into its own repository.

## Improvements

- CUDF PR #472 RMM: Created centralized rmm::device_vector alias and rmm::exec_policy
- CUDF PR #465 Added templated C++ API for RMM to avoid explicit cast to `void**`

## Bug Fixes


RMM was initially implemented as part of cuDF, so we include the relevant changelog history below.

# cuDF 0.3.0 (23 Nov 2018)

## New Features

- PR #336 CSV Reader string support

## Improvements

- CUDF PR #333 Add Rapids Memory Manager documentation
- CUDF PR #321 Rapids Memory Manager adds file/line location logging and convenience macros

## Bug Fixes


# cuDF 0.2.0 and cuDF 0.1.0

These were initial releases of cuDF based on previously separate pyGDF and libGDF libraries. RMM was initially implemented as part of libGDF.
