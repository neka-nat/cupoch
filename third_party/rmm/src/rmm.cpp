/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @brief RAPIDS Memory Manager implementation.
 *
 * Efficient allocation, deallocation and tracking of GPU memory.
 *
 */

#include "rmm/rmm.h"
#include "rmm/detail/memory_manager.hpp"
#include "rmm/mr/cnmem_memory_resource.hpp"
#include "rmm/mr/cnmem_managed_memory_resource.hpp"
#include "rmm/mr/managed_memory_resource.hpp"
#include "rmm/mr/cuda_memory_resource.hpp"



#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstddef>
#include <cuda.h>

#ifndef GETNAME
#define GETNAME(x) case x: return #x;
#endif

// Stringify RMM error code.
const char * rmmGetErrorString(rmmError_t errcode) {
  switch (errcode) {
    // There must be one entry per enum values in gdf_error.
    GETNAME(RMM_SUCCESS)
    GETNAME(RMM_ERROR_CUDA_ERROR)
    GETNAME(RMM_ERROR_INVALID_ARGUMENT)
    GETNAME(RMM_ERROR_NOT_INITIALIZED)
    GETNAME(RMM_ERROR_OUT_OF_MEMORY)
    GETNAME(RMM_ERROR_UNKNOWN)
    GETNAME(RMM_ERROR_IO)
    default:
        // This means we are missing an entry above for a rmmError_t value.
        return "Internal error. Unknown error code.";
  }
}

// Initialize memory manager state and storage.
rmmError_t rmmInitialize(rmmOptions_t *options)
{
  rmm::Manager::getInstance().initialize(options);
  return RMM_SUCCESS;
}

// Shutdown memory manager.
rmmError_t rmmFinalize()
{
  rmm::Manager::getInstance().finalize();
  return RMM_SUCCESS;
}

// Query the initialization state of RMM.
bool rmmIsInitialized(rmmOptions_t *options)
{
  if (nullptr != options) {
    *options = rmm::Manager::getOptions();
  }
  return rmm::Manager::getInstance().isInitialized();
}

// Allocate memory and return a pointer to device memory.
rmmError_t rmmAlloc(void **ptr, size_t size, cudaStream_t stream, const char* file, unsigned int line)
{
  return rmm::alloc(ptr, size, stream, file, line);
}

// Reallocate device memory block to new size and recycle any remaining memory.
//rmmError_t rmmRealloc(void **ptr, size_t new_size, cudaStream_t stream, const char* file, unsigned int line)
//{
//  return rmm::realloc(ptr, new_size, stream, file, line);
//}

// Release device memory and recycle the associated memory.
rmmError_t rmmFree(void *ptr, cudaStream_t stream, const char* file, unsigned int line)
{
  return rmm::free(ptr, stream, file, line);
}

// Get the offset of ptr from its base allocation
rmmError_t rmmGetAllocationOffset(ptrdiff_t *offset,
                                  void *ptr,
                                  cudaStream_t stream)
{
  void *base = (void*)0xffffffff;
  CUresult res = cuMemGetAddressRange((CUdeviceptr*)&base, nullptr,
                                      (CUdeviceptr)ptr);
  if (res != CUDA_SUCCESS)
    return RMM_ERROR_INVALID_ARGUMENT;
  *offset = reinterpret_cast<ptrdiff_t>(ptr) -
            reinterpret_cast<ptrdiff_t>(base);
  return RMM_SUCCESS;
}

// Get amounts of free and total memory managed by a manager associated
// with the stream.
rmmError_t rmmGetInfo(size_t *freeSize, size_t *totalSize, cudaStream_t stream)
{
  try {
    std::pair<size_t,size_t> memInfo = rmm::mr::get_default_resource()->get_mem_info( stream);
    *freeSize = memInfo.first;
    *totalSize = memInfo.second;
  } catch(std::runtime_error const&){
    return RMM_ERROR_CUDA_ERROR;
  }
  return RMM_SUCCESS;
}

// Write the memory event stats log to specified path/filename
rmmError_t rmmWriteLog(const char* filename)
{
  try {
    std::ofstream csv;
    csv.open(filename);
    rmm::Manager::getLogger().to_csv(csv);
  }
  catch (const std::ofstream::failure& e) {
    return RMM_ERROR_IO;
  }
  return RMM_SUCCESS;
}

// Get the size opf the CSV log
size_t rmmLogSize()
{
  std::ostringstream csv;
  rmm::Manager::getLogger().to_csv(csv);
  return csv.str().size();
}

// Get the CSV log as a string
rmmError_t rmmGetLog(char *buffer, size_t buffer_size)
{
  try {
    std::ostringstream csv;
    rmm::Manager::getLogger().to_csv(csv);
    csv.str().copy(buffer, std::min(buffer_size, csv.str().size()));
  }
  catch (const std::ofstream::failure& e) {
    return RMM_ERROR_IO;
  }
  return RMM_SUCCESS;
}
