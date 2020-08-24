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

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/detail/coalescing_free_list.hpp>
#include <rmm/mr/device/detail/stream_ordered_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rmm {
namespace mr {
namespace detail {
/// Gets the available and total device memory in bytes for the current device
inline std::pair<std::size_t, std::size_t> available_device_memory()
{
  std::size_t free{}, total{};
  RMM_CUDA_TRY(cudaMemGetInfo(&free, &total));
  return {free, total};
}
}  // namespace detail

/**
 * @brief A coalescing best-fit suballocator which uses a pool of memory allocated from
 *        an upstream memory_resource.
 *
 * Allocation (do_allocate()) and deallocation (do_deallocate()) are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * @tparam UpstreamResource memory_resource to use for allocating the pool. Implements
 *                          rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class pool_memory_resource final
  : public detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                  detail::coalescing_free_list> {
 public:
  static constexpr size_t default_initial_size = ~0;
  static constexpr size_t default_maximum_size = ~0;
  // TODO use rmm-level def of this.
  static constexpr size_t allocation_alignment = 256;

  friend class detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                      detail::coalescing_free_list>;

  /**
   * @brief Construct a `pool_memory_resource` and allocate the initial device memory pool using
   * `upstream_mr`.
   *
   * @throws rmm::logic_error if `upstream_mr == nullptr`
   *
   * @param upstream_mr The memory_resource from which to allocate blocks for the pool.
   * @param initial_pool_size Minimum size, in bytes, of the initial pool. Defaults to half of the
   * available memory on the current device.
   * @param maximum_pool_size Maximum size, in bytes, that the pool can grow to. Defaults to all of
   * the available memory on the current device.
   */
  explicit pool_memory_resource(Upstream* upstream_mr,
                                std::size_t initial_pool_size = default_initial_size,
                                std::size_t maximum_pool_size = default_maximum_size)
    : upstream_mr_{[upstream_mr]() {
        RMM_EXPECTS(nullptr != upstream_mr, "Unexpected null upstream pointer.");
        return upstream_mr;
      }()}
  {
    std::size_t free{}, total{};
    std::tie(free, total) = detail::available_device_memory();
    initial_pool_size =
      (initial_pool_size == default_initial_size) ? std::min(free, total / 2) : initial_pool_size;
    current_pool_size_ = rmm::detail::align_up(initial_pool_size, allocation_alignment);
    maximum_pool_size_ = (maximum_pool_size == default_maximum_size) ? free : maximum_pool_size;

    RMM_EXPECTS(pool_size() <= maximum_pool_size_,
                "Initial pool size exceeds the maximum pool size!");
    this->insert_block(block_from_upstream(pool_size(), cudaStreamLegacy), cudaStreamLegacy);
  }

  /**
   * @brief Destroy the `pool_memory_resource` and deallocate all memory it allocated using
   * the upstream resource.
   */
  ~pool_memory_resource() { release(); }

  pool_memory_resource()                            = delete;
  pool_memory_resource(pool_memory_resource const&) = delete;
  pool_memory_resource(pool_memory_resource&&)      = delete;
  pool_memory_resource& operator=(pool_memory_resource const&) = delete;
  pool_memory_resource& operator=(pool_memory_resource&&) = delete;

  /**
   * @brief Queries whether the resource supports use of non-null CUDA streams for
   * allocation/deallocation.
   *
   * @returns bool true.
   */
  bool supports_streams() const noexcept override { return true; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool false
   */
  bool supports_get_mem_info() const noexcept override { return false; }

  /**
   * @brief Get the upstream memory_resource object.
   *
   * @return UpstreamResource* the upstream memory resource.
   */
  Upstream* get_upstream() const noexcept { return upstream_mr_; }

 protected:
  using free_list  = detail::coalescing_free_list;
  using block_type = free_list::block_type;
  using typename detail::stream_ordered_memory_resource<pool_memory_resource<Upstream>,
                                                        detail::coalescing_free_list>::split_block;
  using lock_guard = std::lock_guard<std::mutex>;

  /**
   * @brief Get the maximum size of allocations supported by this memory resource
   *
   * Note this does not depend on the memory size of the device. It simply returns the maximum
   * value of `size_t`
   *
   * @return size_t The maximum size of a single allocation supported by this memory resource
   */
  size_t get_maximum_allocation_size() const { return std::numeric_limits<size_t>::max(); }

  /**
   * @brief Allocate space from upstream to supply the suballocation pool and return
   * a sufficiently sized block.
   *
   * @param size The minimum size to allocate
   * @param blocks The free list (ignored in this implementation)
   * @param stream The stream on which the memory is to be used.
   * @return block_type a block of at least `size` bytes
   */
  block_type expand_pool(size_t size, free_list& blocks, cudaStream_t stream)
  {
    auto grow_size = size_to_grow(size);
    RMM_EXPECTS(grow_size > 0, rmm::bad_alloc, "Maximum pool size exceeded");
    current_pool_size_ += grow_size;
    return block_from_upstream(grow_size, stream);
  }

  /**
   * @brief Allocate a block from upstream to expand the suballocation pool.
   *
   * @param size The size in bytes to allocate from the upstream resource
   * @param stream The stream on which the memory is to be used.
   * @return block_type The allocated block
   */
  block_type block_from_upstream(size_t size, cudaStream_t stream)
  {
    void* p = upstream_mr_->allocate(size, stream);
    block_type b{reinterpret_cast<char*>(p), size, true};
    upstream_blocks_.emplace_back(b);  // TODO: with C++17 use version that returns a reference
    return b;
  }

  /**
   * @brief Splits block `b` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned to the pool.
   *
   * @param b The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return A pair comprising the allocated pointer and any unallocated remainder of the input
   * block.
   */
  split_block allocate_from_block(block_type const& b, size_t size)
  {
    block_type const alloc{b.pointer(), size, b.is_head()};
    allocated_blocks_.insert(alloc);

    auto rest =
      (b.size() > size) ? block_type{b.pointer() + size, b.size() - size, false} : block_type{};
    return {reinterpret_cast<void*>(alloc.pointer()), rest};
  }

  /**
   * @brief Finds, frees and returns the block associated with pointer `p`.
   *
   * @param p The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @param stream The stream-event pair for the stream on which the memory was last used.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the pool.
   */
  block_type free_block(void* p, size_t size) noexcept
  {
    if (p == nullptr) return block_type{};

    auto const i = allocated_blocks_.find(static_cast<char*>(p));
    assert(i != allocated_blocks_.end());

    auto block = *i;
    assert(block.size() == rmm::detail::align_up(size, allocation_alignment));
    allocated_blocks_.erase(i);

    return block;
  }

  /**
   * @brief Given a minimum size, computes an appropriate size to grow the pool.
   *
   * Strategy is to try to grow the pool by half the difference between
   * the configured maximum pool size and the current pool size.
   *
   * @param size The size of the minimum allocation immediately needed
   * @return size_t The computed size to grow the pool.
   */
  size_t size_to_grow(size_t size) const
  {
    auto const remaining =
      rmm::detail::align_up(maximum_pool_size_ - pool_size(), allocation_alignment);
    auto const aligned_size = rmm::detail::align_up(size, allocation_alignment);
    if (aligned_size <= remaining / 2) {
      return remaining / 2;
    } else if (aligned_size <= remaining) {
      return aligned_size;
    } else {
      return 0;
    }
  };

  /**
   * @brief Computes the size of the current pool
   *
   * Includes allocated as well as free memory.
   *
   * @return size_t The total size of the currently allocated pool.
   */
  size_t pool_size() const noexcept { return current_pool_size_; }

  /**
   * @brief Free all memory allocated from the upstream memory_resource.
   *
   */
  void release()
  {
    lock_guard lock(this->get_mutex());

    for (auto b : upstream_blocks_)
      upstream_mr_->deallocate(b.pointer(), b.size());
    upstream_blocks_.clear();
    allocated_blocks_.clear();

    current_pool_size_ = 0;
  }

  /**
   * @brief Print debugging information about all blocks in the pool.
   *
   * @note This function is intended only for use in debugging.
   *
   */
  void print()
  {
    lock_guard lock(this->get_mutex());

    std::size_t free, total;
    std::tie(free, total) = upstream_mr_->get_mem_info(0);
    std::cout << "GPU free memory: " << free << " total: " << total << "\n";

    std::cout << "upstream_blocks: " << upstream_blocks_.size() << "\n";
    std::size_t upstream_total{0};

    for (auto h : upstream_blocks_) {
      h.print();
      upstream_total += h.size();
    }
    std::cout << "total upstream: " << upstream_total << " B\n";

    std::cout << "allocated_blocks: " << allocated_blocks_.size() << "\n";
    for (auto b : allocated_blocks_)
      b.print();

    this->print_free_blocks();
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws nothing
   *
   * @param stream to execute on
   * @return std::pair contaiing free_size and total_size of memory
   */
  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const override
  {
    std::size_t free_size{};
    std::size_t total_size{};
    // TODO implement this
    return std::make_pair(free_size, total_size);
  }

  Upstream* upstream_mr_;  // The "heap" to allocate the pool from
  std::size_t current_pool_size_{};
  std::size_t maximum_pool_size_{};

  std::set<block_type, rmm::mr::detail::compare_blocks<block_type>> allocated_blocks_;

  // blocks allocated from upstream: so they can be easily freed
  std::vector<block_type> upstream_blocks_;
};  // namespace mr

}  // namespace mr
}  // namespace rmm
