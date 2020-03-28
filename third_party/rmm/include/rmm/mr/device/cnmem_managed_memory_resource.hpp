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

#include <cnmem.h>
#include "cnmem_memory_resource.hpp"
#include "device_memory_resource.hpp"

#include <cuda_runtime_api.h>
#include <cassert>
#include <exception>
#include <iostream>
#include <mutex>
#include <set>
#include <vector>

namespace rmm {
namespace mr {
/**
 * @brief Memory resource that allocates/deallocates managed device memory
    (CUDA Unified Memory) using the cnmem pool sub-allocator.
 * the cnmem pool sub-allocator for allocation/deallocation.
 */
class cnmem_managed_memory_resource final : public cnmem_memory_resource {
 public:
  /**
   * @brief Construct a cnmem memory resource and allocate the initial device
   * memory pool

   * TODO Add constructor arguments for other CNMEM options/flags
   *
   * @param initial_pool_size Size, in bytes, of the intial pool size. When
   * zero, an implementation defined pool size is used.
   */
  explicit cnmem_managed_memory_resource(std::size_t initial_pool_size = 0,
                                         std::vector<int> const& devices = {})
      : cnmem_memory_resource(initial_pool_size, devices,
                              memory_kind::MANAGED) {}
};

}  // namespace mr
}  // namespace rmm
