/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <rmm/mr/device/detail/free_list.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <list>

namespace rmm {
namespace mr {
namespace detail {

/**
 * @brief A simple block structure specifying the size and location of a block
 *        of memory, with a flag indicating whether it is the head of a block
 *        of memory allocated from the heap (or upstream allocator).
 */
struct block : public block_base {
  block() = default;
  block(char* ptr, size_t size, bool is_head) : block_base{ptr}, size_bytes{size}, head{is_head} {}

  /**
   * @brief Returns the pointer to the memory represented by this block.
   *
   * @return the pointer to the memory represented by this block.
   */
  inline char* pointer() const { return static_cast<char*>(ptr); }

  /**
   * @brief Returns the size of the memory represented by this block.
   *
   * @return the size in bytes of the memory represented by this block.
   */
  inline size_t size() const { return size_bytes; }

  /**
   * @brief Returns whether this block is the start of an allocation from an upstream allocator.
   *
   * A block `b` may not be coalesced with a preceding contiguous block `a` if `b.is_head == true`.
   *
   * @return true if this block is the start of an allocation from an upstream allocator.
   */
  inline bool is_head() const { return head; }

  /**
   * @brief Comparison operator to enable comparing blocks and storing in ordered containers.
   *
   * Orders by ptr address.

   * @param rhs
   * @return true if this block's ptr is < than `rhs` block pointer.
   * @return false if this block's ptr is >= than `rhs` block pointer.
   */
  inline bool operator<(block const& rhs) const noexcept { return pointer() < rhs.pointer(); };

  /**
   * @brief Coalesce two contiguous blocks into one.
   *
   * `this` must immediately precede `b` and both `this` and `b` must be from the same upstream
   * allocation. That is, `this->is_contiguous_before(b)`. Otherwise behavior is undefined.
   *
   * @param b block to merge
   * @return block The merged block
   */
  inline block merge(block const& b) const noexcept
  {
    assert(is_contiguous_before(b));
    return block(pointer(), size() + b.size(), is_head());
  }

  /**
   * @brief Verifies whether this block can be merged to the beginning of block b.
   *
   * @param b The block to check for contiguity.
   * @return true Returns true if this blocks's `ptr` + `size` == `b.ptr`, and `not b.is_head`,
                  false otherwise.
   */
  inline bool is_contiguous_before(block const& b) const noexcept
  {
    return (pointer() + size() == b.ptr) and not(b.is_head());
  }

  /**
   * @brief Is this block large enough to fit `sz` bytes?
   *
   * @param sz The size in bytes to check for fit.
   * @return true if this block is at least `sz` bytes
   */
  inline bool fits(size_t sz) const noexcept { return size() >= sz; }

  /**
   * @brief Is this block a better fit for `sz` bytes than block `b`?
   *
   * @param sz The size in bytes to check for best fit.
   * @param b The other block to check for fit.
   * @return true If this block is a tighter fit for `sz` bytes than block `b`.
   * @return false If this block does not fit `sz` bytes or `b` is a tighter fit.
   */
  inline bool is_better_fit(size_t sz, block const& b) const noexcept
  {
    return fits(sz) && (size() < b.size() || b.size() < sz);
  }

  /**
   * @brief Print this block. For debugging.
   */
  inline void print() const
  {
    std::cout << reinterpret_cast<void*>(pointer()) << " " << size() << " B\n";
  }

 private:
  size_t size_bytes{};  ///< Size in bytes
  bool head{};          ///< Indicates whether ptr was allocated from the heap
};

/// Print block on an ostream
inline std::ostream& operator<<(std::ostream& out, const block& b)
{
  out << b.pointer() << " " << b.size() << " B\n";
  return out;
}

/**
 * @brief Comparator for block types based on pointer address.
 *
 * This comparator allows searching associative containers of blocks by pointer rather than
 * having to search by the contained type. Saves potentially error-prone temporary construction of
 * a block when you just want to search by pointer.
 */
template <typename block_type>
struct compare_blocks {
  // is_transparent (C++14 feature) allows search key type for set<block_type>::find()
  using is_transparent = void;

  bool operator()(block_type const& lhs, block_type const& rhs) const { return lhs < rhs; }
  bool operator()(char const* ptr, block_type const& rhs) const { return ptr < rhs.pointer(); }
  bool operator()(block_type const& lhs, char const* ptr) const { return lhs.pointer() < ptr; };
};

/**
 * @brief An ordered list of free memory blocks that coalesces contiguous blocks on insertion.
 *
 * @tparam list_type the type of the internal list data structure.
 */
struct coalescing_free_list : free_list<block> {
  coalescing_free_list()  = default;
  ~coalescing_free_list() = default;

  /**
   * @brief Inserts a block into the `free_list` in the correct order, coalescing it with the
   *        preceding and following blocks if either is contiguous.
   *
   * @param b The block to insert.
   */
  void insert(block_type const& b)
  {
    if (is_empty()) {
      free_list::insert(cend(), b);
      return;
    }

    // Find the right place (in ascending ptr order) to insert the block
    // Can't use binary_search because it's a linked list and will be quadratic
    auto const next     = std::find_if(begin(), end(), [b](block_type const& i) { return b < i; });
    auto const previous = (next == cbegin()) ? next : std::prev(next);

    // Coalesce with neighboring blocks or insert the new block if it can't be coalesced
    bool const merge_prev = previous->is_contiguous_before(b);
    bool const merge_next = (next != cend()) && b.is_contiguous_before(*next);

    if (merge_prev && merge_next) {
      *previous = previous->merge(b).merge(*next);
      erase(next);
    } else if (merge_prev) {
      *previous = previous->merge(b);
    } else if (merge_next) {
      *next = b.merge(*next);
    } else {
      free_list::insert(next, b);  // cannot be coalesced, just insert
    }
  }

  /**
   * @brief Moves blocks from range `[first, last)` into the free_list in their correct order,
   *        coalescing them with their preceding and following blocks if they are contiguous.
   *
   * @tparam InputIt iterator type
   * @param first The beginning of the range of blocks to insert
   * @param last The end of the range of blocks to insert.
   */
  void insert(free_list&& other)
  {
    std::for_each(std::make_move_iterator(other.begin()),
                  std::make_move_iterator(other.end()),
                  [this](block_type&& b) { this->insert(std::move(b)); });
  }

  /**
   * @brief Finds the smallest block in the `free_list` large enough to fit `size` bytes.
   *
   * This is a "best fit" search.
   *
   * @param size The size in bytes of the desired block.
   * @return block A block large enough to store `size` bytes.
   */
  block_type get_block(size_t size)
  {
    // find best fit block
    auto const iter =
      std::min_element(cbegin(), cend(), [size](block_type const& lhs, block_type const& rhs) {
        return lhs.is_better_fit(size, rhs);
      });

    if (iter != cend() && iter->fits(size)) {
      // Remove the block from the free_list and return it.
      block_type const found = *iter;
      erase(iter);
      return found;
    }

    return block_type{};  // not found
  }

  /**
   * @brief Print all blocks in the free_list.
   */
  void print() const
  {
    std::cout << size() << '\n';
    std::for_each(cbegin(), cend(), [](auto const iter) { iter.print(); });
  }
};  // coalescing_free_list

}  // namespace detail
}  // namespace mr
}  // namespace rmm
