/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#pragma once

#include "device_memory_resource.hpp"

#include <cuda_runtime_api.h>
#include <cassert>
#include <exception>
#include <iostream>

namespace rmm {
namespace mr {
/**---------------------------------------------------------------------------*
 * @brief `device_memory_resource` derived class that uses
 * cudaMallocManaged/Free for allocation/deallocation.
 *---------------------------------------------------------------------------**/
class managed_memory_resource final : public device_memory_resource {
 public:
  bool supports_streams() const noexcept override { return false; }

 private:
  /**--------------------------------------------------------------------------*
   * @brief Allocates memory of size at least \p bytes using cudaMallocManaged.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @note Stream argument is ignored
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   *-------------------------------------------------------------------------**/
  void* do_allocate(std::size_t bytes, cudaStream_t) override {
    // FIXME: Unlike cudaMalloc, cudaMallocManaged will throw an error for 0
    // size allocations.
    if (bytes == 0) {
      return nullptr;
    }

    void* p{nullptr};
    cudaError_t const status = cudaMallocManaged(&p, bytes);
    if (cudaSuccess != status) {
#ifndef NDEBUG
      std::cerr << "cudaMallocManaged failed: " << cudaGetErrorName(status)
                << " " << cudaGetErrorString(status) << "\n";
#endif
      throw std::bad_alloc{};
    }
    return p;
  }

  /**--------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by \p p.
   *
   * @note Stream argument is ignored.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   *-------------------------------------------------------------------------**/
  void do_deallocate(void* p, std::size_t, cudaStream_t) override {
    cudaError_t const status = cudaFree(p);
#ifndef NDEBUG
    if (status != cudaSuccess) {
      std::cerr << "cudaFree failed: " << cudaGetErrorName(status) << " "
                << cudaGetErrorString(status) << "\n";
    }
#endif
  }

  /**--------------------------------------------------------------------------*
   * @brief Compare this resource to another.
   *
   * Two managed_memory_resources always compare equal, because they can each 
   * deallocate memory allocated by the other.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   *-------------------------------------------------------------------------**/
  bool do_is_equal(device_memory_resource const& other) const noexcept {
    return dynamic_cast<managed_memory_resource const*>(&other) != nullptr;
  }

  /**--------------------------------------------------------------------------*
   * @brief Get free and available memory for memory resource
   *
   * @throws std::runtime_error if cudaMemGetInfo fails
   *
   * @param stream to execute on
   * @return std::pair contaiing free_size and total_size of memory
   *-------------------------------------------------------------------------**/
  std::pair<size_t,size_t> do_get_mem_info( cudaStream_t stream) const{
    std::size_t free_size;
    std::size_t total_size;
    auto status = cudaMemGetInfo(&free_size, &total_size);
    if (cudaSuccess != status) {
#ifndef NDEBUG
      std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorName(status) << " "
          << cudaGetErrorString(status) << "\n";
      throw std::runtime_error{
        "Failed to to call get_mem_info on memory resource"};
#endif
    }
    return std::make_pair(free_size, total_size);
  }
};

}  // namespace mr
}  // namespace rmm
