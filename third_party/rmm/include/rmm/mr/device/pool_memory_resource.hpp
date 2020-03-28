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
#include <rmm/mr/device/detail/free_list.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <cassert>
#include <exception>
#include <iostream>
#include <list>
#include <set>
#include <map>
#include <numeric>
#include <algorithm>
#include <mutex>

namespace rmm {
namespace mr {

/**
 * @brief A coalescing best-fit suballocator which uses a pool of memory allocated from
 *        an upstream memory_resource.
 * 
 * @tparam UpstreamResource memory_resource to use for allocating the pool. Implements
 *                          rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class pool_memory_resource final : public device_memory_resource {
 public:

  static constexpr size_t default_initial_size = ~0;
  static constexpr size_t default_maximum_size = ~0;
  // TODO use rmm-level def of this.
  static constexpr size_t allocation_alignment = 256;

  /**
   * @brief Construct a `pool_memory_resource` and allocate the initial
   * device memory pool using `upstream_mr`.
   *
   * @param upstream_mr The memory_resource from which to allocate blocks for the pool.
   * @param initial_pool_size Size, in bytes, of the initial pool. When
   * zero, an implementation-defined pool size is used.
   * @param maximum_pool_size Maximum size, in bytes, that the pool can grow to.
   */
  explicit pool_memory_resource(
      Upstream* upstream_mr,
      std::size_t initial_pool_size = default_initial_size,
      std::size_t maximum_pool_size = default_maximum_size)
      : upstream_mr_{upstream_mr},
        maximum_pool_size_{maximum_pool_size} {
    cudaDeviceProp props;
    int device{0};
    RMM_CUDA_TRY(cudaGetDevice(&device));
    RMM_CUDA_TRY(cudaGetDeviceProperties(&props, device));

    if (initial_pool_size == default_initial_size) {
      initial_pool_size = props.totalGlobalMem / 2;
    }

    initial_pool_size = rmm::detail::align_up(initial_pool_size, allocation_alignment);

    if (maximum_pool_size == default_maximum_size)
        maximum_pool_size_ = props.totalGlobalMem;

    // Allocate initial block
    stream_free_blocks_[0].insert(block_from_upstream(initial_pool_size, 0));
  }

  /**
   * @brief Destroy the `pool_memory_resource` and deallocate all memory it allocated using
   * the upstream resource.
   */
  ~pool_memory_resource() {
    release();
  }

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
  bool supports_get_mem_info() const noexcept override {return false; }

  /**
   * @brief Get the upstream memory_resource object.
   *
   * @return UpstreamResource* the upstream memory resource.
   */
  Upstream* get_upstream() const noexcept { return upstream_mr_; }

 private:

  using block = rmm::mr::detail::block;
  using free_list = rmm::mr::detail::free_list<>;

  /**
   * @brief Find a free block of at least `size` bytes in `free_list` `blocks` associated with
   *        stream `blocks_stream`, for use on `stream`.
   * 
   * @param blocks The `free_list` to look in for a free block of sufficient size.
   * @param blocks_stream The stream that all blocks in `blocks` are associated with.
   * @param size The requested size of the allocation.
   
   * @param stream The stream on which the allocation is being requested.
   * @return block A block with non-null pointer and size >= `size`, or a nullptr block if none is 
   *               available in `blocks`.
   */
  block block_from_stream(free_list& blocks,
                          cudaStream_t blocks_stream,
                          size_t size,
                          cudaStream_t stream) {
    block const b = blocks.best_fit(size); // get the best fit block

    // If we found a block associated with a different stream, 
    // we have to synchronize the stream in order to use it
    if ((blocks_stream != stream) && (b.ptr != nullptr)) { 
      cudaError_t result = cudaStreamSynchronize(blocks_stream);

      RMM_EXPECTS((result == cudaSuccess ||                   // stream synced
                   result == cudaErrorInvalidResourceHandle), // stream deleted
                  rmm::bad_alloc, "cudaStreamSynchronize failure");

      // Now that this stream is synced, insert all other blocks into this stream's list
      // Note: This could cause thrashing between two streams. On the other hand, it reduces
      // fragmentation by coalescing.
      stream_free_blocks_[stream].insert(blocks.begin(), blocks.end());

      // remove this stream from the freelist
      stream_free_blocks_.erase(blocks_stream);
    }
    return b;
  }

  /**
   * @brief Find an available block in the pool of at least `size` bytes, for use on `stream`.
   * 
   * Attempts to find a free block that was last used on `stream` to avoid synchronization. If none
   * is available, it finds a block last used on another stream. In this case, the stream associated
   * with the found block is synchronized to ensure all asynchronous work on the memory is finished
   * before it is used on `stream`.
   * 
   * @param size The size of the requested allocation, in bytes.
   * @param stream The stream on which the allocation will be used.
   * @return block A block with non-null pointer and size >= `size`.
   */
  block available_larger_block(size_t size, cudaStream_t stream) {
    // Try to find a larger block in free list for the same stream
    auto iter = stream_free_blocks_.find(stream);
    if (iter != stream_free_blocks_.end()) {
      block b = block_from_stream(iter->second, stream, size, stream);
      if (b.ptr != nullptr) return b;
    }

    // nothing in this stream's free list, look for one on another stream
    auto s = stream_free_blocks_.begin();
    while (s != stream_free_blocks_.end()) {
      if (s->first != stream) {
        block b = block_from_stream(s->second, s->first, size, stream);
        if (b.ptr != nullptr) return b;
      }
      ++s;
    }

    // no larger blocks available on other streams, so grow the pool and create a block
    size_t grow_size = size_to_grow(size);
    RMM_EXPECTS(grow_size > 0, rmm::bad_alloc, "Maximum pool size exceeded");
    return block_from_upstream(grow_size, stream);
  }

  /**
   * @brief Splits block `b` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned to the pool.
   * 
   * @param b The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @param stream The stream on which the allocation will be used.
   * @return void* The pointer to the allocated memory.
   */
  void* allocate_from_block(block const& b, size_t size, cudaStream_t stream)
  {
    block const alloc{b.ptr, size, b.is_head};

    if (b.size > size)
    {
      block rest{b.ptr + size, b.size - size, false};
      stream_free_blocks_[stream].insert(rest);
    }

    allocated_blocks_.insert(alloc);
    return reinterpret_cast<void*>(alloc.ptr);
  }

  /**
   * @brief Frees the block associated with pointer `p`, returning it to the pool.
   * 
   * @param p The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @param stream The stream on which the memory was last used.
   */
  void free_block(void *p, size_t size, cudaStream_t stream)
  {
    if (p == nullptr) return;

    auto const i = allocated_blocks_.find(block{static_cast<char*>(p)});
    assert(i != allocated_blocks_.end());
    assert(i->size == rmm::detail::align_up(size, allocation_alignment));

    stream_free_blocks_[stream].insert(*i);
    allocated_blocks_.erase(i);
  }

  /**
   * @brief Given a minimum size, computes an appropriate size to grow the pool.
   * 
   * Current strategy is to try to grow the pool by half the difference between
   * the configured maximum pool size and the current pool size.
   * 
   * @param size The size of the minimum allocation immediately needed
   * @return size_t The computed size to grow the pool.
   */
  size_t size_to_grow(size_t size) const {
    auto const remaining = rmm::detail::align_up(maximum_pool_size_ - pool_size(),
                                                 allocation_alignment);
    auto const aligned_size = rmm::detail::align_up(size, allocation_alignment);
    if (aligned_size <= remaining / 2) { return remaining / 2; }
    else if (aligned_size <= remaining) { return remaining; }
    else { return 0; }
  };

  /**
   * @brief Allocates memory of `size` bytes using the upstream memory_resource, on `stream`.
   * 
   * @param size The size of the requested allocation.
   * @param stream The stream on which the requested allocation will be used.
   * @return block A block of at least `size` bytes.
   */
  block block_from_upstream(size_t size, cudaStream_t stream)
  {
    void* p = upstream_mr_->allocate(size, stream);
    block b{reinterpret_cast<char*>(p), size, true};
    upstream_blocks_.emplace_back(b);
    current_pool_size_ += b.size;
    return b;
  }

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
    for (auto b : upstream_blocks_)
      upstream_mr_->deallocate(b.ptr, b.size);
    upstream_blocks_.clear();
    current_pool_size_ = 0;
  }

#ifndef NDEBUG
  /**
   * @brief Print debugging information about all blocks in the pool.
   * 
   */
  void print() {
    std::size_t free, total;
    std::tie(free, total) = upstream_mr_->get_mem_info(0);
    std::cout << "GPU free memory: " << free << "total: " << total << "\n";

    std::cout << "upstream_blocks: " << upstream_blocks_.size() << "\n";
    std::size_t upstream_total{0};

    for (auto h : upstream_blocks_) { 
      h.print();
      upstream_total += h.size;
    }
    std::cout << "total upstream: " << upstream_total << " B\n";

    std::cout << "allocated_blocks: " << allocated_blocks_.size() << "\n";
    for (auto b : allocated_blocks_) { b.print(); }

    std::cout << "sync free blocks: ";
    for (auto s : stream_free_blocks_) {
      std::cout << "stream " << s.first << " ";
      s.second.print();
    }
    std::cout << "\n";
  }
#endif // DEBUG

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @param The stream to associate this allocation with
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    if (bytes <= 0) return nullptr;
    bytes = rmm::detail::align_up(bytes, allocation_alignment);
    block const b = available_larger_block(bytes, stream);
    return allocate_from_block(b, bytes, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws nothing
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    free_block(p, bytes, stream);
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws nothing
   *
   * @param stream to execute on
   * @return std::pair contaiing free_size and total_size of memory
   */
  std::pair<size_t,size_t> do_get_mem_info( cudaStream_t stream) const override {
    std::size_t free_size{};
    std::size_t total_size{};
    // TODO implement this
    return std::make_pair(free_size, total_size);
  }

  size_t maximum_pool_size_;
  size_t current_pool_size_{0};

  Upstream* upstream_mr_;  // The "heap" to allocate the pool from

  // map of [stream_id, free_list] pairs
  // stream stream_id must be synced before allocating from this list to a different stream
  std::map<cudaStream_t, free_list> stream_free_blocks_;

  std::set<block> allocated_blocks_;

  // blocks allocated from upstream: so they can be easily freed
  std::vector<block> upstream_blocks_;
};

}  // namespace mr
}  // namespace rmm
