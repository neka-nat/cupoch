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
/**---------------------------------------------------------------------------*
 * @brief Memory resource that allocates/deallocates using the cnmem pool
 * sub-allocator.
 *---------------------------------------------------------------------------**/
class cnmem_memory_resource : public device_memory_resource {
 public:
  /**--------------------------------------------------------------------------*
   * @brief Construct a cnmem memory resource and allocate the initial device
   * memory pool.

   * TODO Add constructor arguments for other CNMEM options/flags
   *
   * @param initial_pool_size Size, in bytes, of the initial pool size. When
   * zero, an implementation defined pool size is used.
   * @param devices List of GPU device IDs to register with CNMEM
   *---------------------------------------------------------------------------**/
  explicit cnmem_memory_resource(std::size_t initial_pool_size = 0,
                                 std::vector<int> const& devices = {})
      : cnmem_memory_resource(initial_pool_size, devices, memory_kind::CUDA) {}

  virtual ~cnmem_memory_resource() {
    auto status = cnmemFinalize();
#ifndef NDEBUG
    if (status != CNMEM_STATUS_SUCCESS) {
      std::cerr << "cnmemFinalize failed.\n";
    }
#endif
  }

  bool supports_streams() const noexcept override { return true; }

 protected:
  /**
   * @brief The kind of device memory to use for a memory pool
   */
  enum class memory_kind : unsigned short {
    MANAGED,  ///< Uses CUDA managed memory (Unified Memory) for pool
    CUDA      ///< Uses CUDA device memory for pool
  };

  cnmem_memory_resource(std::size_t initial_pool_size,
                        std::vector<int> const& devices,
                        memory_kind pool_type) {
    std::vector<cnmemDevice_t> cnmem_devices;

    // If no devices were specified, use the current one
    if (devices.empty()) {
      int current_device{};
      auto status = cudaGetDevice(&current_device);
      if(status != cudaSuccess){
         throw std::runtime_error{"Failed to get current device."};
      }
      cnmemDevice_t dev{};
      dev.device = current_device;
      dev.size = initial_pool_size;
      cnmem_devices.push_back(dev);
    } else {
      for (auto const& d : devices) {
        cnmemDevice_t dev{};
        dev.device = d;
        dev.size = initial_pool_size;
        cnmem_devices.push_back(dev);
      }
    }

    unsigned long flags =
        (pool_type == memory_kind::MANAGED) ? CNMEM_FLAGS_MANAGED : 0;
    // TODO Update exception
    auto status = cnmemInit(cnmem_devices.size(), cnmem_devices.data(), flags);
    if (CNMEM_STATUS_SUCCESS != status) {
      std::string msg = cnmemGetErrorString(status);
      throw std::runtime_error{"Failed to initialize cnmem: " + msg};
    }
  }

 private:
  /**--------------------------------------------------------------------------*
   * @brief Allocates memory of size at least \p bytes using cnmem.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::runtime_error` if cnmem failed to register the stream
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   *-------------------------------------------------------------------------**/
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    register_stream(stream);
    void* p{nullptr};
    auto status = cnmemMalloc(&p, bytes, stream);
    if (CNMEM_STATUS_SUCCESS != status) {
#ifndef NDEBUG
      std::cerr << "cnmemMalloc failed: " << cnmemGetErrorString(status)
                << "\n";
#endif
      throw std::bad_alloc{};
    }
    return p;
  }

  /**--------------------------------------------------------------------------*
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   *-------------------------------------------------------------------------**/
  void do_deallocate(void* p, std::size_t, cudaStream_t stream) override {
    auto status = cnmemFree(p, stream);
    if (CNMEM_STATUS_SUCCESS != status) {
#ifndef NDEBUG
      std::cerr << "cnmemFree failed: " << cnmemGetErrorString(status) << "\n";
#endif
    }
  }

  void register_stream(cudaStream_t stream) {
    // Don't register null stream with CNMEM
    if (stream != 0) {
      // TODO Probably don't want to have to take a lock for every memory
      // allocation
      std::lock_guard<std::mutex> lock(streams_mutex);
      auto result = registered_streams.insert(stream);

      if (result.second == true) {
        auto status = cnmemRegisterStream(stream);
        if (CNMEM_STATUS_SUCCESS != status) {
          std::string msg = cnmemGetErrorString(status);
          throw std::runtime_error{"Failed to register stream with cnmem: " +
                                   msg};
        }
      }
    }
  }

  /**--------------------------------------------------------------------------*
   * @brief Get free and available memory for memory resource
   *
   * @throws std::runtime_error if we could not get cnmem free / total memory
   *
   * @param stream to execute on
   * @return std::pair contaiing free_size and total_size of memory
   *-------------------------------------------------------------------------**/
  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const {
    std::size_t free_size;
    std::size_t total_size;
    auto status = cnmemMemGetInfo(&free_size, &total_size, stream);
    if (CNMEM_STATUS_SUCCESS != status) {
      std::string msg = cnmemGetErrorString(status);
      throw std::runtime_error{"cnmemMemGetInfo failed: " + msg};
    }
    return std::make_pair(free_size, total_size);
  }

  std::set<cudaStream_t> registered_streams{};
  std::mutex streams_mutex{};
};

}  // namespace mr
}  // namespace rmm
