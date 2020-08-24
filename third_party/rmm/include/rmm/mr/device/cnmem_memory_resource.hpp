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
#include <rmm/detail/error.hpp>
#include "device_memory_resource.hpp"

#include <cuda_runtime_api.h>
#include <cassert>
#include <exception>
#include <iostream>
#include <mutex>
#include <set>
#include <vector>

namespace rmm {
/**
 * @brief Exception thrown when a CNMEM error is encountered.
 *
 */
struct cnmem_error : public std::runtime_error {
  cnmem_error(const char* message) : std::runtime_error(message) {}
  cnmem_error(std::string const& message) : cnmem_error{message.c_str()} {}
};
}  // namespace rmm

#define CNMEM_TRY(...)                                       \
  GET_CNMEM_TRY_MACRO(__VA_ARGS__, CNMEM_TRY_2, CNMEM_TRY_1) \
  (__VA_ARGS__)
#define GET_CNMEM_TRY_MACRO(_1, _2, NAME, ...) NAME
#define CNMEM_TRY_2(_call, _exception_type)                                                        \
  do {                                                                                             \
    cnmemStatus_t const error = (_call);                                                           \
    if (CNMEM_STATUS_SUCCESS != error) {                                                           \
      throw _exception_type{std::string{"CNMEM error at: "} + __FILE__ + RMM_STRINGIFY(__LINE__) + \
                            ": " + cnmemGetErrorString(error)};                                    \
    }                                                                                              \
  } while (0);
#define CNMEM_TRY_1(_call) CNMEM_TRY_2(_call, rmm::cnmem_error)

namespace rmm {
namespace mr {
/**
 * @brief Memory resource that allocates/deallocates using the cnmem pool
 * sub-allocator.
 *
 * @note This class is deprecated as of RMM 0.15. Use pool_memory_resource.
 */
class cnmem_memory_resource : public device_memory_resource {
 public:
  /**
   * @brief Construct a cnmem memory resource and allocate the initial device
   * memory pool.
   *
   * @throws `rmm::cuda_error` if getting the current device fails
   * @throws `rmm::cnmem_error` if initializing CNMEM fails
   *
   * @param initial_pool_size Size, in bytes, of the initial pool size. When
   * zero, an implementation defined pool size is used.
   * @param devices List of GPU device IDs to register with CNMEM
   */
  [[deprecated]] explicit cnmem_memory_resource(std::size_t initial_pool_size   = 0,
                                                std::vector<int> const& devices = {})
    : cnmem_memory_resource(initial_pool_size, devices, memory_kind::CUDA)
  {
  }

  /**
   * @brief Destroy the CNMEM resource, finalizing CNMEM and freeing all pool
   * memory.
   *
   * @throws Nothing.
   *
   */
  virtual ~cnmem_memory_resource()
  {
    auto status = cnmemFinalize();
    assert(CNMEM_STATUS_SUCCESS == status);
  }

  cnmem_memory_resource(cnmem_memory_resource const&) = delete;
  cnmem_memory_resource(cnmem_memory_resource&&)      = delete;
  cnmem_memory_resource& operator=(cnmem_memory_resource const&) = delete;
  cnmem_memory_resource& operator=(cnmem_memory_resource&&) = delete;

  /**
   * @brief Query whether the resource supports use of non-null CUDA streams for
   * allocation/deallocation.
   *
   * @returns bool true
   */
  bool supports_streams() const noexcept override { return true; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return bool true if the resource supports get_mem_info, false otherwise.
   */
  bool supports_get_mem_info() const noexcept override { return true; }

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
                        memory_kind pool_type)
  {
    std::vector<cnmemDevice_t> cnmem_devices;

    // If no devices were specified, use the current one
    if (devices.empty()) {
      int current_device{};
      RMM_CUDA_TRY(cudaGetDevice(&current_device));
      cnmemDevice_t dev{};
      dev.device = current_device;
      dev.size   = initial_pool_size;
      cnmem_devices.push_back(dev);
    } else {
      for (auto const& d : devices) {
        cnmemDevice_t dev{};
        dev.device = d;
        dev.size   = initial_pool_size;
        cnmem_devices.push_back(dev);
      }
    }

    unsigned long flags = (pool_type == memory_kind::MANAGED) ? CNMEM_FLAGS_MANAGED : 0;

    CNMEM_TRY(cnmemInit(cnmem_devices.size(), cnmem_devices.data(), flags));
  }

 private:
  /**
   * @brief Allocates memory of size at least \p bytes using cnmem.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `rmm::cnmem_error` if cnmem failed to register the stream
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override
  {
    register_stream(stream);
    void* p{nullptr};
    CNMEM_TRY(cnmemMalloc(&p, bytes, stream), rmm::bad_alloc);
    return p;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* p, std::size_t, cudaStream_t stream) override
  {
    auto status = cnmemFree(p, stream);
    assert(CNMEM_STATUS_SUCCESS == status);
  }

  /**
   * @brief Registers `stream` with CNMEM if not already registered.
   *
   * The null stream is ignored.
   *
   * If a stream equal to `stream` has already been registered, this function
   * has no effect.
   *
   * This function is thread safe.
   *
   * @throws `rmm::cnmem_error` if registering the stream with cnmem fails.
   *
   * @param stream The stream to register
   */
  void register_stream(cudaStream_t stream)
  {
    // Don't register null stream with CNMEM
    if (stream != 0) {
      // TODO Probably don't want to have to take a lock for every memory
      // allocation
      std::lock_guard<std::mutex> lock(streams_mutex);
      auto result = registered_streams.insert(stream);
      if (result.second == true) { CNMEM_TRY(cnmemRegisterStream(stream)); }
    }
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws rmm::cnmem_error if we could not get cnmem free / total memory
   *
   * @param stream to execute on
   * @return std::pair contaiing free_size and total_size of memory
   **/
  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const override
  {
    std::size_t free_size;
    std::size_t total_size;
    CNMEM_TRY(cnmemMemGetInfo(&free_size, &total_size, stream));
    return std::make_pair(free_size, total_size);
  }
  std::set<cudaStream_t> registered_streams{};  // Unique streams that have been
                                                // registered with cnmem
  std::mutex streams_mutex{};                   // Mutex used to guard concurrent access to
                                                // `registered_streams`
};

}  // namespace mr
}  // namespace rmm
