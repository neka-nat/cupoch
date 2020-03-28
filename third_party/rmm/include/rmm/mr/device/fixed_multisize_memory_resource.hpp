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
#include <rmm/mr/device/fixed_size_memory_resource.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/aligned.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <cuda_runtime_api.h>

#include <list>
#include <unordered_map>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <cassert>
#include <memory>

#include <iostream>
#include "thrust/iterator/transform_iterator.h"

// forward decl
using cudaStream_t = struct CUstream_st*;

namespace rmm {

namespace mr {

namespace {

// Integer pow function
std::size_t ipow(std::size_t base, std::size_t exp) {
  std::size_t ret = 1;
  while (exp > 0) {
    if (exp & 1) ret *= base;  // multiply the result by the current base
    base *= base;    // square the base
    exp = exp >> 1;  // divide the exponent in half
  }
  return ret;
}

} // namespace anonymous


/**
 * @brief Allocates fixed-size memory blocks of a range of sizes.
 * 
 * Allocates blocks in the range `[min_size], max_size]` in power of two steps, 
 * where `min_size` and `max_size` are both powers of two.
 * 
 * @tparam UpstreamResource memory_resource to use for allocating the pool. Implements
 *                          rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class fixed_multisize_memory_resource : public device_memory_resource {
 public:

  // Sizes are 2 << k, for k in [default_min_exponent, default_max_exponent]
  static constexpr std::size_t default_size_base = 2; 
  static constexpr std::size_t default_min_exponent = 18; // 256 KiB
  static constexpr std::size_t default_max_exponent = 22; // 4 MiB

  /**
   * @brief Construct a new fixed multisize memory resource object
   * 
   * Allocates multiple fixed block sizes. The block sizes start at `size_base << min_size_exponent`
   * and grow by powers of `size_base` up to `size_base << max_size_exponent`. So, by default there
   * are 5 block sizes: 2 << 18 (256 KiB), 2 << 19 (512 KiB), 2 << 20 (1 MiB), 2 << 21 (2 MiB), and
   * 2 << 22 (4 MiB).
   * 
   * @throws rmm::logic_error if size_base is not a power of two.
   *
   * @param upstream_resource The upstream memory resource used to allocate pools of blocks
   * @param size_base The base of allocation block sizes, defaults to 2
   * @param min_size_exponent: The exponent of the minimum fixed block size to allocate
   * @param max_size_exponent The exponent of the maximum fixed block size to allocate
   * @param initial_blocks_per_size The number of blocks to preallocate from the upstream memory
   *        resource, and to allocate when all current blocks are in use.
   */
  explicit fixed_multisize_memory_resource(
      Upstream* upstream_resource,
      std::size_t size_base = default_size_base,
      std::size_t min_size_exponent = default_min_exponent,
      std::size_t max_size_exponent = default_max_exponent,
      std::size_t initial_blocks_per_size = 128)
      : upstream_mr_{upstream_resource},
        size_base_{size_base},
        min_size_exponent_{min_size_exponent},
        max_size_exponent_{max_size_exponent},
        min_size_bytes_{ipow(size_base, min_size_exponent)},
        max_size_bytes_{ipow(size_base, max_size_exponent)} {
    RMM_EXPECTS(rmm::detail::is_pow2(size_base), "size_base must be a power of two");
    
    // allocate initial blocks and insert into free list
    for (std::size_t i = min_size_exponent_; i <= max_size_exponent_; i++) {
      fixed_size_mr_.emplace_back(new fixed_size_memory_resource<Upstream>(
        upstream_resource, ipow(size_base, i), initial_blocks_per_size));
    }
  }

  /**
   * @brief Destroy the fixed_multisize_memory_resource and free all memory allocated from the
   *        upstream resource.
   */
  virtual ~fixed_multisize_memory_resource() {}
    
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
  bool supports_get_mem_info() const noexcept override {return false; }

  /**
   * @brief Get the upstream memory_resource object.
   *
   * @return UpstreamResource* the upstream memory resource.
   */
  Upstream* get_upstream() const noexcept { return upstream_mr_; }

  /**
   * @brief Get the minimum block size that this memory_resource can allocate.
   * 
   * @return std::size_t The minimum block size this memory_resource can allocate.
   */
  std::size_t get_min_size() const noexcept { return min_size_bytes_; }

  /**
   * @brief Get the maximum block size that this memory_resource can allocate.
   * 
   * @return std::size_t The maximum block size this memory_resource can allocate.
   */
  std::size_t get_max_size() const noexcept { return max_size_bytes_; }

 private:

  /**
   * @brief Get the memory resource for the requested size
   *
   * Chooses a memory_resource that allocates the smallest blocks at least as large as `bytes`.
   *
   * The behavior is undefined if `bytes` is greater than the maximum block size.
   *
   * @param bytes Requested allocation size in bytes
   * @return rmm::mr::device_memory_resource& memory_resource that can allocate the requested size.
   */
  device_memory_resource* get_resource(std::size_t bytes) {
    assert(bytes <= get_max_size());

    auto exponentiate = [this](std::size_t const& k) { return ipow(size_base_, k); };
    auto min_exp = thrust::make_transform_iterator(
      thrust::make_counting_iterator(min_size_exponent_), exponentiate);
      
    auto iter = std::upper_bound(min_exp, min_exp + (max_size_exponent_ - min_size_exponent_),
      bytes, [](std::size_t const& a, std::size_t const& b) { return a <= b; });

    return fixed_size_mr_[std::distance(min_exp, iter)].get();
  }

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @throws rmm::bad_alloc if size > block_size (constructor parameter)
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    if (bytes <= 0) return nullptr;
    RMM_EXPECTS(bytes <= get_max_size(), rmm::bad_alloc, "bytes must be <= max_size");
    return get_resource(bytes)->allocate(bytes, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws nothing
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    auto res = get_resource(bytes);
    if (res != nullptr) res->deallocate(p, bytes, stream);
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

  Upstream* upstream_mr_;  // The upstream memory_resource from which to allocate blocks.

  std::size_t const size_base_;         // base of the allocation block sizes (power of 2)
  std::size_t const min_size_exponent_; // exponent of the size of smallest blocks allocated
  std::size_t const max_size_exponent_; // exponent of the size of largest blocks allocated
  std::size_t const min_size_bytes_;    // minimum fixed size in bytes
  std::size_t const max_size_bytes_;    // maximum fixed size in bytes

  // allocators for fixed-size blocks <= max_fixed_size_
  std::vector<std::unique_ptr<fixed_size_memory_resource<Upstream>>> fixed_size_mr_;
};
}  // namespace mr
}  // namespace rmm
