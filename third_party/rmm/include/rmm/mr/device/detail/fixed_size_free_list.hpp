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

#include <rmm/mr/device/detail/free_list.hpp>

#include <iostream>

namespace rmm {
namespace mr {
namespace detail {

struct fixed_size_free_list : free_list<block_base> {
  fixed_size_free_list()  = default;
  ~fixed_size_free_list() = default;

  /**
   * @brief Construct a new free_list from range defined by input iterators
   *
   * @tparam InputIt Input iterator
   * @param first The start of the range to insert into the free_list
   * @param last The end of the range to insert into the free_list
   */
  template <class InputIt>
  fixed_size_free_list(InputIt first, InputIt last)
  {
    std::for_each(first, last, [this](block_type const& b) { insert(b); });
  }

  /**
   * @brief Inserts a block into the `free_list` in the correct order, coalescing it with the
   *        preceding and following blocks if either is contiguous.
   *
   * @param b The block to insert.
   */
  void insert(block_type const& b) { push_back(b); }

  /**
   * @brief Splices blocks from range `[first, last)` onto the free_list.
   *
   * @param first The beginning of the range of blocks to insert
   * @param last The end of the range of blocks to insert.
   */
  void insert(free_list&& other) { splice(cend(), std::move(other)); }

  /**
   * @brief Returns the first block in the free list.
   *
   * @param size The size in bytes of the desired block (unused).
   * @return block A block large enough to store `size` bytes.
   */
  block_type get_block(size_t size)
  {
    if (is_empty())
      return block_type{};
    else {
      block_type b = *begin();
      pop_front();
      return b;
    }
  }
};

}  // namespace detail
}  // namespace mr
}  // namespace rmm
