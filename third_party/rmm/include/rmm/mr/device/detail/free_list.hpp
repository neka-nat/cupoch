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

#include <list>
#include <algorithm>
#include <iostream>
#include <cassert>

namespace rmm {
namespace mr {
namespace detail {

/**
 * @brief A simple block structure specifying the size and location of a block
 *        of memory, with a flag indicating whether it is the head of a block
 *        of memory allocated from the heap (or upstream allocator).
 */
struct block
{
  char* ptr;          ///< Raw memory pointer
  size_t size;        ///< Size in bytes
  bool is_head;       ///< Indicates whether ptr was allocated from the heap

  /**
   * @brief Comparison operator to enable comparing blocks and storing in ordered containers.
   * 
   * Orders by ptr address.

   * @param rhs 
   * @return true if this block's ptr is < than `rhs` block pointer.
   * @return false if this block's ptr is >= than `rhs` block pointer.
   */
  bool operator<(block const& rhs) const noexcept { return ptr < rhs.ptr; };

  /**
   * @brief Verifies whether this block can be merged to the beginning of block b.
   * 
   * @param b The block to check for contiguity.
   * @return true Returns true if this blocks's `ptr` + `size` == `b.ptr`, and `not b.is_head`, 
                  false otherwise.
   */
  bool is_contiguous_before(block const& b) const noexcept {
    return (this->ptr + this->size == b.ptr) and not (b.is_head);
  }

  /**
   * @brief Print this block. For debugging.
   */
  void print() const {
    std::cout << reinterpret_cast<void*>(ptr) << " " << size << "B\n";
  }
};

/**
 * @brief Coalesce two contiguous blocks into one.
 *
 * `a` must immediately precede `b` and both `a` and `b` must be from the same upstream allocation.
 * That is, `a.ptr + a.size == b.ptr` and `not b.is_head`. Otherwise behavior is undefined.
 *
 * @param a first block to merge
 * @param b second block to merge
 * @return block The merged block
 */
inline block merge_blocks(block const& a, block const& b)
{
  assert(a.ptr + a.size == b.ptr); // non-contiguous blocks
  assert(not b.is_head); // Can't merge across upstream allocation boundaries

  return block{a.ptr, a.size + b.size};
}

/**
 * @brief An ordered list of free memory blocks that coalesces contiguous blocks on insertion.
 * 
 * @tparam list_type the type of the internal list data structure.
 */
template < typename list_type = std::list<block> >
struct free_list {
  free_list() = default;
  ~free_list() = default;

  using size_type = typename list_type::size_type;
  using iterator = typename list_type::iterator;
  using const_iterator = typename list_type::const_iterator;

  iterator begin() noexcept              { return blocks.begin(); } /// beginning of the free list
  const_iterator begin() const noexcept  { return begin(); }        /// beginning of the free list
  const_iterator cbegin() const noexcept { return blocks.cbegin(); } /// beginning of the free list

  iterator end() noexcept                { return blocks.end(); }   /// end of the free list
  const_iterator end() const noexcept    { return end(); }          /// end of the free list
  const_iterator cend() const noexcept   { return blocks.cend(); }  /// end of the free list

  /**
   * @brief The size of the free list in blocks.
   * 
   * @return size_type The number of blocks in the free list.
   */
  size_type size() const noexcept        { return blocks.size(); }

  /**
   * @brief checks whether the free_list is empty.
   * 
   * @return true If there are blocks in the free_list.
   * @return false If there are no blocks in the free_list.
   */
  bool is_empty() const noexcept         { return blocks.empty(); }

  /**
   * @brief Inserts a block into the `free_list` in the correct order, coalescing it with the
   *        preceding and following blocks if either is contiguous.
   * 
   * @param b The block to insert.
   */
  void insert(block const& b) {
    if (is_empty()) { 
      insert(blocks.end(), b);
      return;
    }

    // Find the right place (in ascending ptr order) to insert the block
    // Can't use binary_search because it's a linked list and will be quadratic
    auto const next = std::find_if(blocks.begin(), blocks.end(),
                             [b](block const& i) { return i.ptr > b.ptr; });
    auto const previous = (next == blocks.begin()) ? next : std::prev(next);

    // Coalesce with neighboring blocks or insert the new block if it can't be coalesced
    bool const merge_prev = previous->is_contiguous_before(b);
    bool const merge_next = (next != blocks.end()) && b.is_contiguous_before(*next);

    if (merge_prev && merge_next) {
      *previous = detail::merge_blocks(*previous, b);
      *previous = detail::merge_blocks(*previous, *next);
      erase(next);
    } else if (merge_prev) {
      *previous = detail::merge_blocks(*previous, b);
    } else if (merge_next) {
      *next = detail::merge_blocks(b, *next);
    } else {
      insert(next, b); // cannot be coalesced, just insert
    }
  }

  /**
   * @brief Inserts blocks from range `[first, last)` into the free_list in their correct order, 
   *        coalescing them with their preceding and following blocks if they are contiguous.
   * 
   * @tparam InputIt iterator type
   * @param first The beginning of the range of blocks to insert
   * @param last The end of the range of blocks to insert.
   */
  template< class InputIt >
  void insert( InputIt first, InputIt last ) {
    std::for_each(first, last, [this](block const& b) { this->insert(b); });
  }

  /**
   * @brief Removes the block indicated by `iter` from the free list.
   * 
   * @param iter An iterator referring to the block to erase.
   */
  void erase(iterator const& iter) {
    blocks.erase(iter);
  }

  /**
  * @brief Erase all blocks from the free_list.
  * 
  */
  void clear() noexcept { blocks.clear(); }

  /**
   * @brief Finds the smallest block in the `free_list` large enough to fit `size` bytes.
   * 
   * @param size The size in bytes of the desired block.
   * @return block A block large enough to store `size` bytes.
   */
  block best_fit(size_t size) {
    // find best fit block
    auto const iter = std::min_element(
      blocks.begin(), blocks.end(), [size](block lhs, block rhs) {
        return (lhs.size >= size) &&
                ((lhs.size < rhs.size) || (rhs.size < size));
      });

    if (iter->size >= size)
    {
      // Remove the block from the free_list and return it.
      block const found = *iter;
      erase(iter);
      return found;
    }
    
    return block{}; // not found
  }

  /**
   * @brief Print all blocks in the free_list.
   */
  void print() const {
    std::cout << blocks.size() << '\n';
    for (block const& b : blocks) { b.print(); }
  }

protected:
  /**
   * @brief Insert a block in the free list before the specified position
   * 
   * @param pos iterator before which the block will be inserted. pos may be the end() iterator.
   * @param b The block to insert.
   */
  void insert(const_iterator pos, block const& b) {
    blocks.insert(pos, b);
  }

private:
  list_type blocks; // The internal container of blocks
}; // free_list

} // namespace rmm::mr::detail
} // namespace rmm::mr
} // namespace rmm
