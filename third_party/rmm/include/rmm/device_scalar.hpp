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

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace rmm {

/**
 * @brief Container for a single object of type `T` in device memory.
 *
 * `T` must be trivially copyable.
 *
 * @tparam T The object's type
 */
template <typename T>
class device_scalar {
 public:
  static_assert(std::is_trivially_copyable<T>::value,
                "Scalar type must be trivially copyable");

  /**
   * @brief Construct a new `device_scalar`
   *
   * @throws `rmm::bad_alloc` if allocating the device memory for
   *`initial_value` fails
   * @throws `rmm::cuda_error` if copying `initial_value` to device memory fails
   *
   * @param initial_value The initial value of the object in device memory
   * @param stream Optional, stream on which to perform allocation and copy
   * @param mr Optional, resource with which to allocate
   */
  explicit device_scalar(
      T const &initial_value, cudaStream_t stream = 0,
      rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
      : buff{sizeof(T), stream, mr} {
    _memcpy(buff.data(), &initial_value, stream);
  }

  /**
   * @brief Copies the value from device to host, synchronizes, and returns the
   * value.
   *
   * @throws `rmm::cuda_error` If the copy fails
   * @throws `rmm::cuda_error` If synchronizing `stream` fails
   *
   * @return T The value of the scalar after synchronizing the stream
   * @param stream CUDA stream on which to perform the copy
   */
  T value(cudaStream_t stream = 0) const {
    T host_value{};
    _memcpy(&host_value, buff.data(), stream);
    return host_value;
  }

  /**
   * @brief Copies the value from host to device and synchronizes the stream.
   *
   * @throws `rmm::cuda_error` if copying `host_value` to device memory fails
   * @throws `rmm::cuda_error` if synchronizing `stream` fails
   *
   * @param host_value The host value which will be copied to device
   * @param stream CUDA stream on which to perform the copy
   */
  void set_value(T host_value, cudaStream_t stream = 0) {
    _memcpy(buff.data(), &host_value, stream);
  }

  /**
   * @brief Returns pointer to object in device memory.
   */
  T *data() noexcept { return static_cast<T *>(buff.data()); }

  /**
   * @brief Returns pointer to object in device memory.
   */
  T const *data() const noexcept { return static_cast<T const *>(buff.data()); }

  device_scalar() = default;
  ~device_scalar() = default;
  device_scalar(device_scalar const &) = default;
  device_scalar(device_scalar &&) = default;
  device_scalar &operator=(device_scalar const &) = delete;
  device_scalar &operator=(device_scalar &&) = delete;

 private:
  rmm::device_buffer buff{sizeof(T)};

  inline void _memcpy(void *dst, const void *src, cudaStream_t stream) const {
    RMM_CUDA_TRY(cudaMemcpyAsync(dst, src, sizeof(T), cudaMemcpyDefault, stream));
    RMM_CUDA_TRY(cudaStreamSynchronize(stream));
  }
};
}  // namespace rmm
