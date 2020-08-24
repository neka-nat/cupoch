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

#include <cstddef>
#include <rmm/detail/aligned.hpp>
#include <utility>

// forward decl
using cudaStream_t = struct CUstream_st*;

namespace rmm {

namespace mr {
/**
 * @brief Base class for all libcudf device memory allocation.
 *
 * This class serves as the interface that all custom device memory
 * implementations must satisfy.
 *
 * There are two private, pure virtual functions that all derived classes must implement:
 *`do_allocate` and `do_deallocate`. Optionally, derived classes may also override `is_equal`. By
 * default, `is_equal` simply performs an identity comparison.
 *
 * The public, non-virtual functions `allocate`, `deallocate`, and `is_equal` simply call the
 * private virtual functions. The reason for this is to allow implementing shared, default behavior
 * in the base class. For example, the base class' `allocate` function may log every allocation, no
 * matter what derived class implementation is used.
 *
 * The `allocate` and `deallocate` APIs and implementations provide stream-ordered memory
 * allocation. This allows optimizations such as re-using memory deallocated on the same stream
 * without the overhead of stream synchronization.
 *
 * A call to `allocate(bytes, stream_a)` (on any derived class) returns a pointer that is valid to
 * use on `stream_a`. Using the memory on a different stream (say `stream_b`) is Undefined Behavior
 * unless the two streams are first synchronized, for example by using
 * `cudaStreamSynchronize(stream_a)` or by recording a CUDA event on `stream_a` and then
 * calling `cudaStreamWaitEvent(stream_b, event)`.
 *
 * The stream specified to deallocate() should be a stream on which it is valid to use the
 * deallocated memory immediately for another allocation. Typically this is the stream on which the
 * allocation was *last* used before the call to deallocate(). The passed stream may be used
 * internally by a device_memory_resource for managing available memory with minimal
 * synchronization, and it may also be synchronized at a later time, for example using a call to
 * `cudaStreamSynchronize()`.
 *
 * For this reason, it is Undefined Behavior to destroy a CUDA stream that is passed to
 * deallocate(). If the stream on which the allocation was last used has been destroyed before
 * calling deallocate() or it is known that it will be destroyed, it is likely better to synchronize
 * the stream (before destroying it) and then pass a different stream to deallocate() (e.g. the
 * default stream).
 *
 * A device_memory_resource should only be used when the active CUDA device is the same device
 * that was active when the device_memory_resource was created. Otherwise behavior is undefined.
 *
 * Creating a device_memory_resource for each device requires care to set the current device
 * before creating each resource, and to maintain the lifetime of the resources as long as they
 * are set as per-device resources. Here is an example loop that creates `unique_ptr`s to
 * pool_memory_resource objects for each device and sets them as the per-device resource for that
 * device.
 *
 * @code{c++}
 * std::vector<unique_ptr<pool_memory_resource>> per_device_pools;
 * for(int i = 0; i < N; ++i) {
 *   cudaSetDevice(i);
 *   per_device_pools.push_back(std::make_unique<pool_memory_resource>());
 *   set_per_device_resource(cuda_device_id{i}, &per_device_pools.back());
 * }
 * @endcode
 */
class device_memory_resource {
 public:
  virtual ~device_memory_resource() = default;

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @throws `rmm::bad_alloc` When the requested `bytes` cannot be allocated on
   * the specified `stream`.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* allocate(std::size_t bytes, cudaStream_t stream = 0)
  {
    return do_allocate(rmm::detail::align_up(bytes, 8), stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * `p` must have been returned by a prior call to `allocate(bytes,stream)` on
   * a `device_memory_resource` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void deallocate(void* p, std::size_t bytes, cudaStream_t stream = 0)
  {
    do_deallocate(p, rmm::detail::align_up(bytes, 8), stream);
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two device_memory_resources compare equal if and only if memory allocated
   * from one device_memory_resource can be deallocated from the other and vice
   * versa.
   *
   * By default, simply checks if \p *this and \p other refer to the same
   * object, i.e., does not check if they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @returns If the two resources are equivalent
   */
  bool is_equal(device_memory_resource const& other) const noexcept { return do_is_equal(other); }

  /**
   * @brief Query whether the resource supports use of non-null CUDA streams for
   * allocation/deallocation.
   *
   * @returns bool true if the resource supports non-null CUDA streams.
   */
  virtual bool supports_streams() const noexcept = 0;

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool true if the resource supports get_mem_info, false otherwise.
   */
  virtual bool supports_get_mem_info() const noexcept = 0;

  /**
   * @brief Queries the amount of free and total memory for the resource.
   *
   * @param stream the stream whose memory manager we want to retrieve
   *
   * @returns a std::pair<size_t,size_t> which contains free memory in bytes
   * in .first and total amount of memory in .second
   */
  std::pair<std::size_t, std::size_t> get_mem_info(cudaStream_t stream) const
  {
    return do_get_mem_info(stream);
  }

 private:
  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  virtual void* do_allocate(std::size_t bytes, cudaStream_t stream) = 0;

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * If supported, this operation may optionally be executed on a stream.
   * Otherwise, the stream is ignored and the null stream is used.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  virtual void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) = 0;

  /**
   * @brief Compare this resource to another.
   *
   * Two device_memory_resources compare equal if and only if memory allocated
   * from one device_memory_resource can be deallocated from the other and vice
   * versa.
   *
   * By default, simply checks if \p *this and \p other refer to the same
   * object, i.e., does not check if they are two objects of the same class.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  virtual bool do_is_equal(device_memory_resource const& other) const noexcept
  {
    return this == &other;
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws std::runtime_error if we could not get free / total memory
   *
   * @param stream the stream being executed on
   * @return std::pair with available and free memory for resource
   */
  virtual std::pair<std::size_t, std::size_t> do_get_mem_info(cudaStream_t stream) const = 0;
};
}  // namespace mr
}  // namespace rmm
