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

#include "host_memory_resource.hpp"

#include <rmm/detail/aligned.hpp>

#include <cstddef>
#include <utility>

namespace rmm {
namespace mr {

/**---------------------------------------------------------------------------*
 * @brief A `host_memory_resource` that uses the global `operator new` and
 * `operator delete` to allocate host memory.
 *---------------------------------------------------------------------------**/
class new_delete_resource final : public host_memory_resource {
 public:
  new_delete_resource()                            = default;
  ~new_delete_resource()                           = default;
  new_delete_resource(new_delete_resource const &) = default;
  new_delete_resource(new_delete_resource &&)      = default;
  new_delete_resource &operator=(new_delete_resource const &) = default;
  new_delete_resource &operator=(new_delete_resource &&) = default;

 private:
  /**---------------------------------------------------------------------------*
   * @brief Allocates memory on the host of size at least `bytes` bytes.
   *
   * The returned storage is aligned to the specified `alignment` if supported,
   * and to `alignof(std::max_align_t)` otherwise.
   *
   * @throws std::bad_alloc When the requested `bytes` and `alignment` cannot be
   * allocated.
   *
   * @param bytes The size of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   *---------------------------------------------------------------------------**/
  void *do_allocate(std::size_t bytes,
                    std::size_t alignment = detail::RMM_DEFAULT_HOST_ALIGNMENT) override
  {
#if __cplusplus >= 201703L
    return ::operator new(bytes, std::align_val_t(alignment));
#else

    // If the requested alignment isn't supported, use default
    alignment =
      (detail::is_supported_alignment(alignment)) ? alignment : detail::RMM_DEFAULT_HOST_ALIGNMENT;

    return detail::aligned_allocate(
      bytes, alignment, [](std::size_t size) { return ::operator new(size); });
#endif
  }

  /**---------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by `p`.
   *
   * `p` must have been returned by a prior call to `allocate(bytes,alignment)`
   * on a `host_memory_resource` that compares equal to `*this`, and the storage
   * it points to must not yet have been deallocated, otherwise behavior is
   * undefined.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param alignment Alignment of the allocation. This must be equal to the
   *value of `alignment` that was passed to the `allocate` call that returned
   *`p`.
   * @param stream Stream on which to perform deallocation
   *---------------------------------------------------------------------------**/
  void do_deallocate(void *p,
                     std::size_t bytes,
                     std::size_t alignment = detail::RMM_DEFAULT_HOST_ALIGNMENT) override
  {
#if __cplusplus >= 201703L
    ::operator delete(p, bytes, std::align_val_t(alignment));
#else
    detail::aligned_deallocate(p, bytes, alignment, [](void *p) { ::operator delete(p); });
#endif
  }
};
}  // namespace mr
}  // namespace rmm
