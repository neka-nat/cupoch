# RMM 0.12.0 (Date TBD)

## New Features

## Improvements

- PR #214 Add codeowners

## Bug Fixes
- PR #215 Catch polymorphic exceptions by reference instead of by value


# RMM 0.11.0 (Date TBD)

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

