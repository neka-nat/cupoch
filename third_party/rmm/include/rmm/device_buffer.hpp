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

#include <rmm/detail/error.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <cuda_runtime_api.h>
#include <cassert>
#include <stdexcept>
#include <utility>

namespace rmm {
/**
 * @file device_buffer.hpp
 * @brief RAII construct for device memory allocation
 *
 * This class allocates untyped and *uninitialized* device memory using a
 * `device_memory_resource`. If not explicitly specified, the memory resource
 * returned from `get_default_resource()` is used.
 *
 * @note Unlike `std::vector` or `thrust::device_vector`, the device memory
 * allocated by a `device_buffer` is uninitialized. Therefore, it is undefined
 * behavior to read the contents of `data()` before first initializing it.
 *
 * Examples:
 * ```
 * //Allocates at least 100 bytes of device memory using the default memory
 * //resource and default stream.
 * device_buffer buff(100);
 *
 * // allocates at least 100 bytes using the custom memory resource and
 * // specified stream
 * custom_memory_resource mr;
 * cudaStream_t stream = 0;
 * device_buffer custom_buff(100, stream, &mr);
 *
 * // deep copies `buff` into a new device buffer using the default stream
 * device_buffer buff_copy(buff);
 *
 * // deep copies `buff` into a new device buffer using the specified stream
 * device_buffer buff_copy(buff, stream);
 *
 * // shallow copies `buff` into a new device_buffer, `buff` is now empty
 * device_buffer buff_move(std::move(buff));
 *
 * // Default construction. Buffer is empty
 * device_buffer buff_default{};
 *
 * // If the requested size is larger than the current size, resizes allocation
 * // to the new size and deep copies any previous contents. Otherwise, simply
 * // updates the value of `size()` to the newly requested size without any
 * // allocations or copies. Uses the optionally specified stream or the default
 * // stream if none specified.
 * buff_default.resize(100, stream);
 *```
 */
class device_buffer {
 public:
  /**
   * @brief Default constructor creates an empty `device_buffer`
   */
  // Note: we cannot use `device_buffer() = default;` because nvcc implicitly adds
  // `__host__ __device__` specifiers to the defaulted constructor when it is called within the 
  // context of both host and device functions. Specifically, the `cudf::type_dispatcher` is a host-
  // device function. This causes warnings/errors because this ctor invokes host-only functions.
  device_buffer()
      : _data{nullptr},
        _size{},
        _capacity{},
        _stream{},
        _mr{rmm::mr::get_default_resource()} {}


  /**
   * @brief Constructs a new device buffer of `size` uninitialized bytes
   *
   * @throws rmm::bad_alloc If allocation fails.
   *
   * @param size Size in bytes to allocate in device memory.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation.
   */
  explicit device_buffer(
      std::size_t size, cudaStream_t stream = 0,
      mr::device_memory_resource* mr = mr::get_default_resource())
      : _stream{stream}, _mr{mr} {
    allocate(size);
  }

  /**
   * @brief Construct a new device buffer by copying from a raw pointer to an
   * existing host or device memory allocation.
   *
   * @throws rmm::bad_alloc If creating the new allocation fails.
   * @throws rmm::logic_error If `source_data` is null, and `size != 0`.
   * @throws rmm::cuda_error if copying from the device memory fails.
   *
   * @param source_data Pointer to the host or device memory to copy from.
   * @param size Size in bytes to copy.
   * @param stream CUDA stream on which memory may be allocated if the memory
   * resource supports streams.
   * @param mr Memory resource to use for the device memory allocation
   */
  device_buffer(void const* source_data, std::size_t size,
                cudaStream_t stream = 0,
                mr::device_memory_resource* mr = mr::get_default_resource())
      : _stream{stream}, _mr{mr} {
    allocate(size);
    copy(source_data, size);
  }

  /**
   * @brief Construct a new `device_buffer` by deep copying the contents of
   * another `device_buffer`, optionally using the specified stream and memory
   * resource.
   *
   * @note Only copies `other.size()` bytes from `other`, i.e., if
   *`other.size() != other.capacity()`, then the size and capacity of the newly
   * constructed `device_buffer` will be equal to `other.size()`.
   *
   * @throws rmm::bad_alloc If creating the new allocation fails.
   * @throws rmm::cuda_error if copying from `other` fails.
   *
   * @param other The `device_buffer` whose contents will be copied
   * @param stream The stream to use for the allocation and copy
   * @param mr The resource to use for allocating the new `device_buffer`
   */
  device_buffer(
      device_buffer const& other, cudaStream_t stream = 0,
      rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource())
      : device_buffer{other.data(), other.size(), stream, mr} {}

  /**
   * @brief Constructs a new `device_buffer` by moving the contents of another
   * `device_buffer` into the newly constructed one.
   *
   * After the new `device_buffer` is constructed, `other` is modified to be a
   * valid, empty `device_buffer`, i.e., `data()` returns `nullptr`, and
   * `size()` and `capacity()` are zero.
   *
   * @throws Nothing
   *
   * @param other The `device_buffer` whose contents will be moved into the
   * newly constructed one.
   */
  device_buffer(device_buffer&& other) noexcept
      : _data{other._data},
        _size{other._size},
        _capacity{other._capacity},
        _stream{other.stream()},
        _mr{other._mr} {
    other._data = nullptr;
    other._size = 0;
    other._capacity = 0;
    other.set_stream(0);
  }

  /**
   * @brief Copies the contents of `other` into this `device_buffer`.
   *
   * All operations on the data in this `device_buffer` on all streams must be
   * complete before using this operator, otherwise behavior is undefined.
   *
   * If the existing capacity is large enough, and the memory resources are
   * compatible, then this `device_buffer`'s existing memory will be reused and
   * `other`s contents will simply be copied on `other.stream()`. I.e., if
   * `capcity() > other.size()` and
   * `memory_resource()->is_equal(*other.memory_resource())`.
   *
   * Otherwise, the existing memory will be deallocated using
   * `memory_resource()` on `stream()` and new memory will be allocated using
   * `other.memory_resource()` on `other.stream()`.
   *
   * @throws rmm::bad_alloc if allocation fails
   * @throws rmm::cuda_error if the copy from `other` fails
   *
   * @param other The `device_buffer` to copy.
   */
  device_buffer& operator=(device_buffer const& other) {
    if (&other != this) {
      // If the current capacity is large enough and the resources are
      // compatible, just reuse the existing memory
      if ((capacity() > other.size()) and _mr->is_equal(*other._mr)) {
        set_stream(other.stream());
        resize(other.size());
        copy(other.data(), other.size());
      } else {
        // Otherwise, need to deallocate and allocate new memory
        deallocate();
        set_stream(other.stream());
        _mr = other._mr;
        allocate(other.size());
        copy(other.data(), other.size());
      }
    }
    return *this;
  }

  /**
   * @brief Move assignment operator moves the contents from `other`.
   *
   * This `device_buffer`'s current device memory allocation will be deallocated
   * on `stream()`.
   *
   * If a different stream is required, call `set_stream()` on
   * the instance before assignment. After assignment, this instance's stream is
   * replaced by the `other.stream()`.
   *
   * @param other The `device_buffer` whose contents will be moved.
   */
  device_buffer& operator=(device_buffer&& other) noexcept {
    if (&other != this) {
      deallocate();

      _data = other._data;
      _size = other._size;
      _capacity = other._capacity;
      set_stream(other.stream());
      _mr = other._mr;

      other._data = nullptr;
      other._size = 0;
      other._capacity = 0;
      other.set_stream(0);
    }
    return *this;
  }

  /**
   * @brief Destroy the device buffer object
   *
   * @note If the memory resource supports streams, this destructor deallocates
   * using the stream most recently passed to any of this device buffer's
   * methods.
   */
  ~device_buffer() noexcept {
    deallocate();
    _mr = nullptr;
    _stream = 0;
  }

  /**
   * @brief Resize the device memory allocation
   *
   * If the requested `new_size` is less than or equal to `capacity()`, no
   * action is taken other than updating the value that is returned from
   * `size()`. Specifically, no memory is allocated nor copied. The value
   * `capacity()` remains the actual size of the device memory allocation.
   *
   * @note `shrink_to_fit()` may be used to force the deallocation of unused
   * `capacity()`.
   *
   * If `new_size` is larger than `capacity()`, a new allocation is made on
   * `stream` to satisfy `new_size`, and the contents of the old allocation are
   * copied on `stream` to the new allocation. The old allocation is then freed.
   * The bytes from `[old_size, new_size)` are uninitialized.
   *
   * The invariant `size() <= capacity()` holds.
   *
   * @throws rmm::bad_alloc If creating the new allocation fails
   * @throws rmm::cuda_error if the copy from the old to new allocation
   *fails
   *
   * @param new_size The requested new size, in bytes
   * @param stream The stream to use for allocation and copy
   */
  void resize(std::size_t new_size, cudaStream_t stream = 0) {
    set_stream(stream);
    // If the requested size is smaller than the current capacity, just update
    // the size without any allocations
    if (new_size <= capacity()) {
      _size = new_size;
    } else {
      void* const new_data = _mr->allocate(new_size, this->stream());
      RMM_CUDA_TRY(cudaMemcpyAsync(new_data, data(), size(), cudaMemcpyDefault,
                                   this->stream()));
      deallocate();
      _data = new_data;
      _size = new_size;
      _capacity = new_size;
    }
  }

  /**
   * @brief Forces the deallocation of unused memory.
   *
   * Reallocates and copies on stream `stream` the contents of the device memory
   * allocation to reduce `capacity()` to `size()`.
   *
   * If `size() == capacity()`, no allocations nor copies occur.
   *
   * @throws rmm::bad_alloc If creating the new allocation fails
   * @throws rmm::cuda_error If the copy from the old to new allocation fails
   *
   * @param stream The stream on which the allocation and copy are performed
   */
  void shrink_to_fit(cudaStream_t stream = 0) {
    set_stream(stream);
    if (size() != capacity()) {
      // Invoke copy ctor on self which only copies `[0, size())` and swap it
      // with self. The temporary `device_buffer` will hold the old contents
      // which will then be destroyed
      auto tmp = device_buffer{*this, stream};
      std::swap(tmp, *this);
    }
  }

  /**
   * @brief Returns raw pointer to underlying device memory allocation
   */
  void const* data() const noexcept { return _data; }

  /**
   * @brief Returns raw pointer to underlying device memory allocation
   */
  void* data() noexcept { return _data; }

  /**
   * @brief Returns size in bytes that was requested for the device memory
   * allocation
   */
  std::size_t size() const noexcept { return _size; }

  /**
   * @brief Returns whether the size in bytes of the `device_buffer` is zero.
   *
   * If `is_empty() == true`, the `device_buffer` may still hold an allocation
   * if `capacity() > 0`.
   *
   */
  bool is_empty() const noexcept { return 0 == size(); }

  /**
   * @brief Returns actual size in bytes of device memory allocation.
   *
   * The invariant `size() <= capacity()` holds.
   */
  std::size_t capacity() const noexcept { return _capacity; }

  /**
   * @brief Returns stream most recently specified for allocation/deallocation
   */
  cudaStream_t stream() const noexcept { return _stream; }

  /**
   * @brief Sets the stream to be used for deallocation
   *
   * If no other rmm::device_buffer method that allocates or copies memory is
   * called after this call with a different stream argument, then @p stream
   * will be used for deallocation in the `rmm::device_buffer destructor.
   * Otherwise, if another rmm::device_buffer method with a stream parameter is
   * called after this, the later stream parameter will be stored and used in
   * the destructor.
   */
  void set_stream(cudaStream_t stream) noexcept { _stream = stream; }

  /**
   * @brief Returns pointer to the memory resource used to allocate and
   * deallocate the device memory
   */
  mr::device_memory_resource* memory_resource() const noexcept { return _mr; }

 private:
  void* _data{nullptr};     ///< Pointer to device memory allocation
  std::size_t _size{};      ///< Requested size of the device memory allocation
  std::size_t _capacity{};  ///< The actual size of the device memory allocation
  cudaStream_t _stream{};   ///< Stream to use for device memory deallocation
  mr::device_memory_resource* _mr{
      mr::get_default_resource()};  ///< The memory resource used to
                                    ///< allocate/deallocate device memory

  /**
   * @brief Allocates the specified amount of memory and updates the
   * size/capacity accordingly.
   *
   * If `bytes == 0`, sets `_data = nullptr`.
   *
   * @param bytes The amount of memory to allocate
   * @param stream The stream on which to allocate
   */
  void allocate(std::size_t bytes) {
    _size = bytes;
    _capacity = bytes;
    _data = (bytes > 0) ? _mr->allocate(bytes, stream()) : nullptr;
  }

  /**
   * @brief Deallocate any memory held by this `device_buffer` and clear the
   * size/capacity/data members.
   *
   * If the buffer doesn't hold any memory, i.e., `capacity() == 0`, doesn't
   * call the resource deallocation.
   *
   */
  void deallocate() noexcept {
    if (capacity() > 0) {
      _mr->deallocate(data(), capacity());
    }
    _size = 0;
    _capacity = 0;
    _data = nullptr;
  }

  /**
   * @brief Copies the specified number of `bytes` from `source` into the
   * internal device allocation.
   *
   * `source` can point to either host or device memory.
   *
   * This function assumes `_data` already points to an allocation large enough
   * to hold `bytes` bytes.
   *
   * @param source The pointer to copy from
   * @param bytes The number of bytes to copy
   */
  void copy(void const* source, std::size_t bytes) {
    if (bytes > 0) {
      RMM_EXPECTS(nullptr != source, "Invalid copy from nullptr.");

      RMM_CUDA_TRY(
          cudaMemcpyAsync(_data, source, bytes, cudaMemcpyDefault, stream()));
    }
  }
};
}  // namespace rmm
