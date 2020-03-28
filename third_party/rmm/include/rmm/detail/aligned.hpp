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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>

namespace rmm {
namespace detail {

/**
 * @brief Default alignment used for host memory allocated by RMM.
 *
 */
static constexpr std::size_t RMM_DEFAULT_HOST_ALIGNMENT{
    alignof(std::max_align_t)};

/**
 * @brief Returns whether or not `n` is a power of 2.
 *
 */
constexpr bool is_pow2(std::size_t n) { return (0 == (n & (n - 1))); }

/**
 * @brief Returns whether or not `alignment` is a valid memory alignment.
 *
 */
constexpr bool is_supported_alignment(std::size_t alignment) {
  return is_pow2(alignment);
}

/**
 * @brief Allocates sufficient memory to satisfy the requested size `bytes` with
 * alignment `alignment` using the unary callable `alloc` to allocate memory.
 *
 * Given a pointer `p` to an allocation of size `n` returned from the unary
 * callable `alloc`, the pointer `q` returned from `aligned_alloc` points to a
 * location within the `n` bytes with sufficient space for `bytes` that
 * satisfies `alignment`.
 *
 * In order to retrieve the original allocation pointer `p`, the offset
 * between `p` and `q` is stored at `q - sizeof(std::ptrdiff_t)`.
 *
 * Allocations returned from `aligned_allocate` *MUST* be freed by calling
 * `aligned_deallocate` with the same arguments for `bytes` and `alignment` with
 * a compatible unary `dealloc` callable capable of freeing the memory returned
 * from `alloc`.
 *
 * If `alignment` is not a power of 2, behavior is undefined.
 *
 * @param bytes The desired size of the allocation
 * @param alignment Desired alignment of allocation
 * @param alloc Unary callable given a size `n` will allocate at least `n` bytes
 * of host memory.
 * @tparam Alloc a unary callable type that allocates memory.
 * @return void* Pointer into allocation of at least `bytes` with desired
 * `alignment`.
 */
template <typename Alloc>
void *aligned_allocate(std::size_t bytes, std::size_t alignment, Alloc alloc) {
  assert(is_pow2(alignment));

  // allocate memory for bytes, plus potential alignment correction,
  // plus store of the correction offset
  std::size_t padded_allocation_size{bytes + alignment +
                                     sizeof(std::ptrdiff_t)};

  char *const original = static_cast<char *>(alloc(padded_allocation_size));

  // account for storage of offset immediately prior to the aligned pointer
  void *aligned{original + sizeof(std::ptrdiff_t)};

  // std::align modifies `aligned` to point to the first aligned location
  std::align(alignment, bytes, aligned, padded_allocation_size);

  // Compute the offset between the original and aligned pointers
  std::ptrdiff_t offset = static_cast<char *>(aligned) - original;

  // Store the offset immediately before the aligned pointer
  *(static_cast<std::ptrdiff_t *>(aligned) - 1) = offset;

  return aligned;
}

/**
 * @brief Frees an allocation returned from `aligned_allocate`.
 *
 * Allocations returned from `aligned_allocate` *MUST* be freed by calling
 * `aligned_deallocate` with the same arguments for `bytes` and `alignment`
 * with a compatible unary `dealloc` callable capable of freeing the memory
 * returned from `alloc`.
 *
 * @param p The aligned pointer to deallocate
 * @param bytes The number of bytes requested from `aligned_allocate`
 * @param alignment The alignment required from `aligned_allocate`
 * @param dealloc A unary callable capable of freeing memory returned from
 * `alloc` in `aligned_allocate`.
 * @tparam Dealloc A unary callable type that deallocates memory.
 */
template <typename Dealloc>
void aligned_deallocate(void *p, std::size_t bytes, std::size_t alignment,
                        Dealloc dealloc) {
  (void)alignment;

  // Get offset from the location immediately prior to the aligned pointer
  std::ptrdiff_t const offset = *(reinterpret_cast<std::ptrdiff_t *>(p) - 1);

  void *const original = static_cast<char *>(p) - offset;

  dealloc(original);
}
}  // namespace detail
}  // namespace rmm
