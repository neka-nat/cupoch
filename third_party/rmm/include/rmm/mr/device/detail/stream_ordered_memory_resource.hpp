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

#include <limits>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>

namespace rmm {
namespace mr {
namespace detail {

/**
 * @brief A CRTP helper function
 *
 * https://www.fluentcpp.com/2017/05/19/crtp-helper/
 *
 * Does two things:
 * 1. Makes "crtp" explicit in the inheritance structure of a CRTP base class.
 * 2. Avoids having to `static_cast` in a lot of places
 *
 * @tparam T The derived class in a CRTP hierarchy
 */
template <typename T>
struct crtp {
  T& underlying() { return static_cast<T&>(*this); }
  T const& underlying() const { return static_cast<T const&>(*this); }
};

/**
 * @brief Base class for a stream-ordered memory resource
 *
 * This base class uses CRTP (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
 * to provide static polymorphism to enable defining suballocator resources that maintain separate
 * pools per stream. All of the stream-ordering logic is contained in this class, but the logic
 * to determine how memory pools are managed and the type of allocation is implented in a derived
 * class and in a free list class.
 *
 * For example, a coalescing pool memory resource uses a coalescing_free_list and maintains data
 * structures for allocated blocks and has functions to allocate and free blocks and to expand the
 * pool.
 *
 * Classes derived from stream_ordered_memory_resource must implement the following four methods,
 * documented separately:
 *
 * 1. `size_t get_maximum_allocation_size() const`
 * 2. `block_type expand_pool(size_t size, free_list& blocks, cudaStream_t stream)`
 * 3. `split_block allocate_from_block(block_type const& b, size_t size)`
 * 4. `block_type free_block(void* p, size_t size) noexcept`
 */
template <typename PoolResource, typename FreeListType>
class stream_ordered_memory_resource : public crtp<PoolResource>, public device_memory_resource {
 public:
  // TODO use rmm-level def of this.
  static constexpr size_t allocation_alignment = 256;

  ~stream_ordered_memory_resource() { release(); }

  stream_ordered_memory_resource()                                      = default;
  stream_ordered_memory_resource(stream_ordered_memory_resource const&) = delete;
  stream_ordered_memory_resource(stream_ordered_memory_resource&&)      = delete;
  stream_ordered_memory_resource& operator=(stream_ordered_memory_resource const&) = delete;
  stream_ordered_memory_resource& operator=(stream_ordered_memory_resource&&) = delete;

 protected:
  using free_list  = FreeListType;
  using block_type = typename free_list::block_type;
  using lock_guard = std::lock_guard<std::mutex>;

  // Derived classes must implement these four methods

  /**
   * @brief Get the maximum size of a single allocation supported by this suballocator memory
   * resource
   *
   * Default implementation is the maximum `size_t` value, but fixed-size allocators will have a
   * lower limit. Override this function in derived classes as necessary.
   *
   * @return size_t The maximum size of a single allocation supported by this memory resource
   */
  // size_t get_maximum_allocation_size() const { return std::numeric_limits<size_t>::max(); }

  /**
   * @brief Allocate space (typically from upstream) to supply the suballocation pool and return
   * a sufficiently sized block.
   *
   * This function returns a block because in some suballocators, a single block is allocated
   * from upstream and returned. In other suballocators, many blocks are created from upstream. In
   * the latter case, the function returns one block and inserts all the rest into the free list
   * `blocks`.
   *
   * @param size The minimum size block to return
   * @param blocks The free list into which to optionally insert new blocks
   * @param stream The stream on which the memory is to be used.
   * @return block_type a block of at least `size` bytes
   */
  // block_type expand_pool(size_t size, free_list& blocks, cudaStream_t stream)

  /// Struct representing a block that has been split for allocation
  struct split_block {
    void* allocated_pointer;  ///< The pointer allocated from a block
    block_type remainder;     ///< The remainder of the block from which the pointer was allocated
  };

  /**
   * @brief Split block `b` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned as the remainder element in the output
   * `split_block`.
   *
   * @param b The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return A `split_block` comprising the allocated pointer and any unallocated remainder of the
   * input block.
   */
  // split_block allocate_from_block(block_type const& b, size_t size)

  /**
   * @brief Finds, frees and returns the block associated with pointer `p`.
   *
   * @param p The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `p`. The caller is expected to return the block
   * to the pool.
   */
  // block_type free_block(void* p, size_t size) noexcept

  /**
   * @brief Returns the block `b` (last used on stream `stream_event`) to the pool.
   *
   * @param b The block to insert into the pool.
   * @param stream The stream on which the memory was last used.
   */
  void insert_block(block_type const& b, cudaStream_t stream)
  {
    stream_free_blocks_[get_event(stream)].insert(b);
  }

  void insert_blocks(free_list&& blocks, cudaStream_t stream)
  {
    stream_free_blocks_[get_event(stream)].insert(std::move(blocks));
  }

  void print_free_blocks() const
  {
    std::cout << "stream free blocks: ";
    for (auto s : stream_free_blocks_) {
      std::cout << "stream: " << s.first.stream << " event: " << s.first.event << " ";
      s.second.print();
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  /**
   * @brief Get the mutex object
   *
   * @return std::mutex
   */
  std::mutex& get_mutex() { return mtx_; }

  struct stream_event_pair {
    cudaStream_t stream;
    cudaEvent_t event;

    bool operator<(stream_event_pair const& rhs) const { return event < rhs.event; }
  };

  /**
   * @brief Allocates memory of size at least `bytes`.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size in bytes of the allocation
   * @param stream The stream to associate this allocation with
   * @return void* Pointer to the newly allocated memory
   */
  virtual void* do_allocate(std::size_t bytes, cudaStream_t stream) override
  {
    if (bytes <= 0) return nullptr;

    lock_guard lock(mtx_);

    auto stream_event = get_event(stream);
    bytes             = rmm::detail::align_up(bytes, allocation_alignment);
    RMM_EXPECTS(bytes <= this->underlying().get_maximum_allocation_size(),
                rmm::bad_alloc,
                "Maximum allocation size exceeded");
    auto const b = this->underlying().get_block(bytes, stream_event);
    auto split   = this->underlying().allocate_from_block(b, bytes);
    if (split.remainder.is_valid()) stream_free_blocks_[stream_event].insert(split.remainder);
    return split.allocated_pointer;
  }

  /**
   * @brief Deallocate memory pointed to by `p`.
   *
   * @throws nothing
   *
   * @param p Pointer to be deallocated
   */
  virtual void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override
  {
    lock_guard lock(mtx_);
    auto stream_event = get_event(stream);
    bytes             = rmm::detail::align_up(bytes, allocation_alignment);
    auto const b      = this->underlying().free_block(p, bytes);

    // TODO: cudaEventRecord has significant overhead on deallocations. For the non-PTDS case
    // we may be able to delay recording the event in some situations. But using events rather than
    // streams allows stealing from deleted streams.
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(stream_event.event, stream));

    stream_free_blocks_[stream_event].insert(b);
  }

 private:
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  /**
   * @brief RAII wrapper for a CUDA event.
   */
  struct event_wrapper {
    event_wrapper()
    {
      RMM_ASSERT_CUDA_SUCCESS(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    ~event_wrapper() { RMM_ASSERT_CUDA_SUCCESS(cudaEventDestroy(event)); }
    cudaEvent_t event{};
  };

// using default_stream_event_type =
//  default_stream_event<stream_ordered_suballocator_memory_resource<FreeListType>>;
#endif

  /**
   * @brief get a unique CUDA event (possibly new) associated with `stream`
   *
   * The event is created on the first call, and it is not recorded. If compiled for per-thread
   * default stream and `stream` is the default stream, the event is created in thread local
   * memory and is unique per CPU thread.
   *
   * @param stream The stream for which to get an event.
   * @return The stream_event for `stream`.
   */
  stream_event_pair get_event(cudaStream_t stream)
  {
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    if (cudaStreamDefault == stream || cudaStreamPerThread == stream) {
      // Create a thread-local shared event wrapper. Shared pointers in the thread and in each MR
      // instance ensures it is destroyed cleaned up only after all are finished with it.
      thread_local auto event_tls = std::make_shared<event_wrapper>();
      default_stream_events.insert(event_tls);
      return stream_event_pair{stream, event_tls.get()->event};
    }
#else
    // We use cudaStreamLegacy as the event map key for the default stream for consistency between
    // PTDS and non-PTDS mode. In PTDS mode, the cudaStreamLegacy map key will only exist if the
    // user explicitly passes it, so it is used as the default location for the free list
    // at construction. For consistency, the same key is used for null stream free lists in non-PTDS
    // mode.
    if (cudaStreamDefault == stream) { stream = cudaStreamLegacy; }
#endif

    auto iter = stream_events_.find(stream);
    return (iter != stream_events_.end()) ? iter->second : [&]() {
      stream_event_pair stream_event{stream};
      RMM_ASSERT_CUDA_SUCCESS(
        cudaEventCreateWithFlags(&stream_event.event, cudaEventDisableTiming));
      stream_events_[stream] = stream_event;
      return stream_event;
    }();
  }

  /**
   * @brief Get an avaible memory block of at least `size` bytes
   *
   * @param size The number of bytes to allocate
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return block_type A block of memory of at least `size` bytes
   */
  block_type get_block(size_t size, stream_event_pair stream_event)
  {
    // Try to find a satisfactory block in free list for the same stream (no sync required)
    auto iter = stream_free_blocks_.find(stream_event);
    if (iter != stream_free_blocks_.end()) {
      block_type b = iter->second.get_block(size);
      if (b.is_valid()) return b;
    }

    // Otherwise try to find a block associated with another stream
    block_type b = get_block_from_other_stream(size, stream_event);
    if (b.is_valid()) return b;

    // no larger blocks available on other streams, so grow the pool and create a block

    // avoid searching for this stream's list again
    free_list& blocks =
      (iter != stream_free_blocks_.end()) ? iter->second : stream_free_blocks_[stream_event];

    return this->underlying().expand_pool(size, blocks, stream_event.stream);
  }

  /**
   * @brief Find a free block of at least `size` bytes in a `free_list` with a different
   * stream/event than `stream_event`.
   *
   * If an appropriate block is found in a free list F associated with event E, if
   * `CUDA_API_PER_THREAD_DEFAULT_STREAM` is defined, `stream_event.stream` will be made to wait
   * on event E. Otherwise, the stream associated with free list F will be synchronized. In either
   * case all other blocks in free list F will be moved to the free list associated with
   * `stream_event.stream`. This results in coalescing with other blocks in that free list,
   * hopefully reducing fragmentation.
   *
   * @param size The requested size of the allocation.
   * @param stream_event The stream and associated event on which the allocation is being
   * requested.
   * @return A block with non-null pointer and size >= `size`, or a nullptr block if none is
   *         available in `blocks`.
   */
  block_type get_block_from_other_stream(size_t size, stream_event_pair stream_event)
  {
    // nothing in this stream's free list, look for one on another stream
    for (auto s = stream_free_blocks_.begin(); s != stream_free_blocks_.end(); ++s) {
      auto blocks_event = s->first;
      if (blocks_event.event != stream_event.event) {
        auto blocks = s->second;

        block_type const b = blocks.get_block(size);  // get the best fit block

        if (b.is_valid()) {
          // Since we found a block associated with a different stream, we have to insert a wait
          // on the stream's associated event into the allocating stream.
          RMM_CUDA_TRY(cudaStreamWaitEvent(stream_event.stream, blocks_event.event, 0));

          // Move all the blocks to the requesting stream, since it has waited on them
          stream_free_blocks_[stream_event].insert(std::move(blocks));
          stream_free_blocks_.erase(s);

          return b;
        }
      }
    }
    return block_type{};
  }

  /**
   * @brief Clear free lists and events
   *
   * Note: only called by destructor.
   */
  void release()
  {
    lock_guard lock(mtx_);

    for (auto s_e : stream_events_) {
      RMM_ASSERT_CUDA_SUCCESS(cudaEventSynchronize(s_e.second.event));
      RMM_ASSERT_CUDA_SUCCESS(cudaEventDestroy(s_e.second.event));
    }

    stream_events_.clear();
    stream_free_blocks_.clear();
  }

  // map of stream_event_pair --> free_list
  // Event (or associated stream) must be synced before allocating from associated free_list to a
  // different stream
  std::map<stream_event_pair, free_list> stream_free_blocks_;

  // bidirectional mapping between non-default streams and events
  std::unordered_map<cudaStream_t, stream_event_pair> stream_events_;

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  // shared pointers to events keeps the events alive as long as either the thread that created them
  // or the MR that is using them exists.
  std::set<std::shared_ptr<event_wrapper>> default_stream_events;
#endif

  std::mutex mtx_;  // mutex for thread-safe access
};                  // namespace detail

}  // namespace detail
}  // namespace mr
}  // namespace rmm
