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
#include <rmm/mr/device/detail/fixed_size_free_list.hpp>
#include <rmm/mr/device/detail/stream_ordered_memory_resource.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <list>
#include <map>
#include <utility>
#include <vector>

namespace rmm {

namespace mr {

/**
 * @brief A `device_memory_resource` which allocates memory blocks of a single fixed size.
 *
 * Supports only allocations of size smaller than the configured block_size.
 */
template <typename Upstream>
class fixed_size_memory_resource
  : public detail::stream_ordered_memory_resource<fixed_size_memory_resource<Upstream>,
                                                  detail::fixed_size_free_list> {
 public:
  friend class detail::stream_ordered_memory_resource<fixed_size_memory_resource<Upstream>,
                                                      detail::fixed_size_free_list>;

  // A block is the fixed size this resource alloates
  static constexpr std::size_t default_block_size = 1 << 20;  // 1 MiB
  // This is the number of blocks that the pool starts out with, and also the number of
  // blocks by which the pool grows when all of its current blocks are allocated
  static constexpr std::size_t default_blocks_to_preallocate = 128;
  // The required alignment of this allocator
  static constexpr std::size_t allocation_alignment = 256;

  /**
   * @brief Construct a new `fixed_size_memory_resource` that allocates memory from
   * `upstream_resource`.
   *
   * When the pool of blocks is all allocated, grows the pool by allocating
   * `blocks_to_preallocate` more blocks from `upstream_mr`.
   *
   * @param upstream_mr The memory_resource from which to allocate blocks for the pool.
   * @param block_size The size of blocks to allocate.
   * @param blocks_to_preallocate The number of blocks to allocate to initialize the pool.
   */
  explicit fixed_size_memory_resource(
    Upstream* upstream_mr,
    std::size_t block_size            = default_block_size,
    std::size_t blocks_to_preallocate = default_blocks_to_preallocate)
    : upstream_mr_{upstream_mr},
      block_size_{rmm::detail::align_up(block_size, allocation_alignment)},
      upstream_chunk_size_{block_size * blocks_to_preallocate}
  {
    // allocate initial blocks and insert into free list
    this->insert_blocks(std::move(blocks_from_upstream(cudaStreamLegacy)), cudaStreamLegacy);
  }

  /**
   * @brief Destroy the `fixed_size_memory_resource` and free all memory allocated from upstream.
   *
   */
  ~fixed_size_memory_resource() { release(); }

  fixed_size_memory_resource()                                  = delete;
  fixed_size_memory_resource(fixed_size_memory_resource const&) = delete;
  fixed_size_memory_resource(fixed_size_memory_resource&&)      = delete;
  fixed_size_memory_resource& operator=(fixed_size_memory_resource const&) = delete;
  fixed_size_memory_resource& operator=(fixed_size_memory_resource&&) = delete;

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
   * @brief Get the upstream memory_resource object.
   *
   * @return UpstreamResource* the upstream memory resource.
   */
  Upstream* get_upstream() const noexcept { return upstream_mr_; }

  /**
   * @brief Get the size of blocks allocated by this memory resource.
   *
   * @return std::size_t size in bytes of allocated blocks.
   */
  std::size_t get_block_size() const noexcept { return block_size_; }

 protected:
  using free_list  = detail::fixed_size_free_list;
  using block_type = free_list::block_type;
  using typename detail::stream_ordered_memory_resource<fixed_size_memory_resource<Upstream>,
                                                        detail::fixed_size_free_list>::split_block;
  using lock_guard = std::lock_guard<std::mutex>;

  /**
   * @brief Get the (fixed) size of allocations supported by this memory resource
   *
   * @return size_t The (fixed) maximum size of a single allocation supported by this memory
   * resource
   */
  size_t get_maximum_allocation_size() const { return get_block_size(); }

  /**
   * @brief Allocate a block from upstream to supply the suballocation pool.
   *
   * Note typically the allocated size will be larger than requested, and is based on the growth
   * strategy (see `size_to_grow()`).
   *
   * @param size The minimum size to allocate
   * @param stream The stream on which the memory is to be used.
   * @return block_type The allocated block
   */
  block_type expand_pool(size_t size, free_list& blocks, cudaStream_t stream)
  {
    blocks.insert(std::move(blocks_from_upstream(stream)));
    return blocks.get_block(size);
  }

  /**
   * @brief Allocate blocks from upstream to expand the suballocation pool.
   *
   * @param size The minimum size to allocate
   * @param stream The stream on which the memory is to be used.
   * @return block_type The allocated block
   */
  free_list blocks_from_upstream(cudaStream_t stream)
  {
    void* p = upstream_mr_->allocate(upstream_chunk_size_, stream);
    block_type b{p};
    upstream_blocks_.push_back(b);

    auto num_blocks = upstream_chunk_size_ / block_size_;

    auto g     = [p, this](int i) { return block_type{static_cast<char*>(p) + i * block_size_}; };
    auto first = thrust::make_transform_iterator(thrust::make_counting_iterator(std::size_t{0}), g);
    return free_list(first, first + num_blocks);
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
    return split_block{b.pointer(), block_type{nullptr}};
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
    // Deallocating a fixed-size block just inserts it in the free list, which is
    // handled by the parent class
    assert(rmm::detail::align_up(size, allocation_alignment) <= block_size_);
    return block_type{p};
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws std::runtime_error if we could not get free / total memory
   *
   * @param stream the stream being executed on
   * @return std::pair with available and free memory for resource
   */
  std::pair<std::size_t, std::size_t> do_get_mem_info(cudaStream_t stream) const override
  {
    return std::make_pair(0, 0);
  }

  /**
   * @brief free all memory allocated using the upstream resource.
   *
   */
  void release()
  {
    lock_guard lock(this->get_mutex());

    for (auto b : upstream_blocks_)
      upstream_mr_->deallocate(b.pointer(), upstream_chunk_size_);
    upstream_blocks_.clear();
  }

#ifndef NDEBUG
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
      upstream_total += upstream_chunk_size_;
    }
    std::cout << "total upstream: " << upstream_total << " B\n";

    this->print_free_blocks();
  }
#endif

  Upstream* upstream_mr_;  // The resource from which to allocate new blocks

  std::size_t const block_size_;           // size of blocks this MR allocates
  std::size_t const upstream_chunk_size_;  // size of chunks allocated from heap MR

  // blocks allocated from heap: so they can be easily freed
  std::vector<block_type> upstream_blocks_;
};

}  // namespace mr
}  // namespace rmm
