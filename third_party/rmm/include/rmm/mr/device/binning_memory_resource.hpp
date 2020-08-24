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

#include <rmm/detail/aligned.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/fixed_size_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <map>
#include <memory>
#include <vector>

namespace rmm {
namespace mr {

/**
 * @brief Allocates memory from upstream resources associated with bin sizes.
 *
 * @tparam UpstreamResource memory_resource to use for allocations that don't fall within any
 * configured bin size. Implements rmm::mr::device_memory_resource interface.
 */
template <typename Upstream>
class binning_memory_resource final : public device_memory_resource {
 public:
  // The required alignment of this allocator
  static constexpr std::size_t allocation_alignment = 256;

  /**
   * @brief Construct a new binning memory resource object.
   *
   * Initially has no bins, so simply uses the upstream_resource until bin resources are added
   * with `add_bin`.
   *
   * @throws rmm::logic_error if size_base is not a power of two.
   *
   * @param upstream_resource The upstream memory resource used to allocate bin pools.
   */
  explicit binning_memory_resource(Upstream* upstream_resource)
    : upstream_mr_{[upstream_resource]() {
        RMM_EXPECTS(nullptr != upstream_resource, "Unexpected null upstream pointer.");
        return upstream_resource;
      }()}
  {
  }

  /**
   * @brief Construct a new binning memory resource object with a range of initial bins.
   *
   * Constructs a new binning memory resource and adds bins backed by `fixed_size_memory_resource`
   * in the range [2^min_size_exponent, 2^max_size_exponent]. For example if `min_size_exponent==18`
   * and `max_size_exponent==22`, creates bins of sizes 256KiB, 512KiB, 1024KiB, 2048KiB and
   * 4096KiB.
   *
   * @param upstream_resource The upstream memory resource used to allocate bin pools.
   * @param min_size_exponent The minimum base-2 exponent bin size.
   * @param max_size_exponent The maximum base-2 exponent bin size.
   */
  binning_memory_resource(Upstream* upstream_resource,
                          int8_t min_size_exponent,
                          int8_t max_size_exponent)
    : upstream_mr_{[upstream_resource]() {
        RMM_EXPECTS(nullptr != upstream_resource, "Unexpected null upstream pointer.");
        return upstream_resource;
      }()}
  {
    for (auto i = min_size_exponent; i <= max_size_exponent; i++)
      add_bin(1 << i);
  }

  /**
   * @brief Destroy the binning_memory_resource and free all memory allocated from the upstream
   * resource.
   */
  ~binning_memory_resource() = default;

  binning_memory_resource()                               = delete;
  binning_memory_resource(binning_memory_resource const&) = delete;
  binning_memory_resource(binning_memory_resource&&)      = delete;
  binning_memory_resource& operator=(binning_memory_resource const&) = delete;
  binning_memory_resource& operator=(binning_memory_resource&&) = delete;

  /**
   * @brief Query whether the resource supports use of non-null streams for
   * allocation/deallocation.
   *
   * @returns true
   */
  bool supports_streams() const noexcept override { return true; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool true if the resource supports get_mem_info, false otherwise.
   */
  bool supports_get_mem_info() const noexcept override { return false; }

  /**
   * @brief Get the upstream memory_resource object.
   *
   * @return UpstreamResource* the upstream memory resource.
   */
  Upstream* get_upstream() const noexcept { return upstream_mr_; }

  /**
   * @brief Add a bin allocator to this resource
   *
   * Adds `bin_resource` if it is not null; otherwise constructs and adds a
   * fixed_size_memory_resource.
   *
   * This bin will be used for any allocation smaller than `allocation_size` that is larger than
   * the next smaller bin's allocation size.
   *
   * If there is already a bin of the specified size nothing is changed.
   *
   * This function is not thread safe.
   *
   * @param allocation_size The maximum size that this bin allocates
   * @param bin_resource The memory resource for the bin
   */
  void add_bin(std::size_t allocation_size, device_memory_resource* bin_resource = nullptr)
  {
    allocation_size = rmm::detail::align_up(allocation_size, allocation_alignment);

    if (nullptr != bin_resource)
      resource_bins_.insert({allocation_size, bin_resource});
    else {
      // If the bin already exists, do nothing.
      if (resource_bins_.count(allocation_size) == 0) {
        owned_bin_resources_.push_back(
          std::make_unique<fixed_size_memory_resource<Upstream>>(upstream_mr_, allocation_size));
        resource_bins_.insert({allocation_size, owned_bin_resources_.back().get()});
      }
    }
  }

 private:
  /**
   * @brief Get the memory resource for the requested size
   *
   * Chooses a memory_resource that allocates the smallest blocks at least as large as `bytes`.
   *
   * @param bytes Requested allocation size in bytes
   * @return rmm::mr::device_memory_resource& memory_resource that can allocate the requested size.
   */
  device_memory_resource* get_resource(std::size_t bytes)
  {
    auto iter = resource_bins_.lower_bound(bytes);
    return (iter != resource_bins_.cend()) ? iter->second
                                           : static_cast<device_memory_resource*>(get_upstream());
  }

  /**
   * @brief Allocates memory of size at least \p bytes.
   *
   * The returned pointer will have at minimum 256 byte alignment.
   *
   * @param bytes The size of the allocation
   * @param stream Stream on which to perform allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override
  {
    if (bytes <= 0) return nullptr;
    return get_resource(bytes)->allocate(bytes, stream);
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws nothing
   *
   * @param p Pointer to be deallocated
   * @param bytes The size in bytes of the allocation. This must be equal to the
   * value of `bytes` that was passed to the `allocate` call that returned `p`.
   * @param stream Stream on which to perform deallocation
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override
  {
    auto res = get_resource(bytes);
    if (res != nullptr) res->deallocate(p, bytes, stream);
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws std::runtime_error if we could not get free / total memory
   *
   * @param stream the stream being executed on
   * @return std::pair with available and free memory for resource
   */
  std::pair<std::size_t, std::size_t> do_get_mem_info(cudaStream_t stream) const override
  {
    return std::make_pair(0, 0);
  }

  Upstream* upstream_mr_;  // The upstream memory_resource from which to allocate blocks.

  std::vector<std::unique_ptr<fixed_size_memory_resource<Upstream>>> owned_bin_resources_;

  std::map<std::size_t, device_memory_resource*> resource_bins_;
};

}  // namespace mr
}  // namespace rmm
