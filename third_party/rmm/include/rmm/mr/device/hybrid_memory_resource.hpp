/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <list>
#include <memory>
#include <unordered_map>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <cassert>


// forward decl
using cudaStream_t = struct CUstream_st*;

namespace rmm {

namespace mr {

/**
 * @brief Allocates memory from one of two allocators selected based on the requested size.
 *
 * For example, using a fixed_multisize_memory_resource for allocations below the size
 * threshold, and a pool_memory_resource for allocations above the size threshold. This can be 
 * advantageous if there are many small allocations, because fixed_multisize_memory_resource is much 
 * faster than pool_memory_resource, and it can reduce fragmentation that can impact the
 * availability of large contiguous blocks of memory.
 *
 * @tparam SmallAllocMemoryResource the memory_resource type to use for small allocations.
 * @tparam LargeAllocMemoryResource the memory_resource type to use for large allocations.
 */
 template <typename SmallAllocMemoryResource, typename LargeAllocMemoryResource>
class hybrid_memory_resource : public device_memory_resource {
 public:

  // The default size used to select between the two allocators.
  static constexpr std::size_t default_threshold_size = 1 << 22; // 4 MiB

  /**
   * @brief Construct a new hybrid_memory_resource_object that selects between the two
   * specified memory_resources based on the `threshold_size`.
   * 
   * @param small_mr The memory_resource to use for small allocations.
   * @param large_mr The memory_resource to use for large allocations.
   * @param threshold_size Allocations > this size (in bytes) use large_mr. Allocations <= this size
   *                       use small_mr.
   */
  hybrid_memory_resource(
    SmallAllocMemoryResource* small_mr,
    LargeAllocMemoryResource* large_mr,
    std::size_t threshold_size = default_threshold_size)
  : small_mr_{small_mr}, large_mr_{large_mr}, threshold_size_{threshold_size} {
  }

  /**
   * @brief Destroy the hybrid-memory_resource object.
   * 
   * @note since hybrid_memory_resource does not own its upstream memory_resources, this does not
   *       free any memory.
   */
  virtual ~hybrid_memory_resource() = default;

  /**
   * @brief Query whether the resource supports use of non-null streams for
   * allocation/deallocation.
   *
   * @returns true
   */
  bool supports_streams() const noexcept override { return true; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   * 
   * @return bool true if the resource supports get_mem_info, false otherwise.
   */
  bool supports_get_mem_info() const noexcept override { return false; }

  /**
   * @brief Get the upstream memory_resource used for small allocations.
   * 
   * @return SmallAllocMemoryResource* the upstream resource used for small allocations.
   */
  SmallAllocMemoryResource* get_small_mr() { return small_mr_; }
  /**
   * @brief Get the upstream memory_resource used for large allocations.
   * 
   * @return LargeAllocMemoryResource* the upstream resource used for large allocations.
   */
  LargeAllocMemoryResource* get_large_mr() { return large_mr_; }

 private:

  /**
   * @brief Get the memory resource to use to allocate the requested size.
   * 
   * Returns the small memory_resource if `bytes` is <= the threshold size, or the large
   * memory_resource otherwise.
   *
   * @param bytes Requested allocation size in bytes.
   * @return rmm::mr::device_memory_resource& memory_resource that can allocate the requested size.
   */
  rmm::mr::device_memory_resource* get_resource(std::size_t bytes) {
    if (bytes <= threshold_size_) return small_mr_;
    else return large_mr_;
  }

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    if (bytes <= 0) return nullptr;
    return get_resource(bytes)->allocate(bytes, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    get_resource(bytes)->deallocate(p, bytes, stream);
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws std::runtime_error if we could not get free / total memory
   *
   * @param stream the stream being executed on
   * @return std::pair with available and free memory for resource
   */
  std::pair<std::size_t, std::size_t> do_get_mem_info( cudaStream_t stream) const override {
    return std::make_pair(0, 0);
  }

  std::size_t const threshold_size_;  // threshold for choosing memory_resource

  SmallAllocMemoryResource *small_mr_; // allocator for <= max_small_size
  LargeAllocMemoryResource *large_mr_; // allocator for > max_small_size
};
}  // namespace mr
}  // namespace rmm
