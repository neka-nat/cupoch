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

#include <rmm/mr/device/device_memory_resource.hpp>

#include <rmm/detail/error.hpp>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <memory>

namespace rmm {
namespace mr {
/**
 * @brief Resource that uses `Upstream` to allocate memory and logs information
 * about the requested allocation/deallocations.
 *
 * An instance of this resource can be constructured with an existing, upstream
 * resource in order to satisfy allocation requests and log
 * allocation/deallocation activity.
 *
 * @tparam Upstream Type of the upstream resource used for
 * allocation/deallocation.
 */
template <typename Upstream>
class logging_resource_adaptor final : public device_memory_resource {
 public:
  /**
   * @brief Construct a new logging resource adaptor using `upstream` to satisfy
   * allocation requests and logging information about each allocation/free to
   * the file specified by `filename`.
   *
   * The logfile will be written using CSV formatting.
   *
   * Clears the contents of `filename` if it already exists.
   *
   * Creating multiple `logging_resource_adaptor`s with the same `filename` will
   * result in undefined behavior.
   *
   * @throws `rmm::logic_error` if `upstream == nullptr`
   * @throws `spdlog::spdlog_ex` if opening `filename` failed
   *
   * @param upstream The resource used for allocating/deallocating device memory
   * @param filename Name of file to write log info. If not specified, retrieves
   * the file name from the environment variable "RMM_LOG_FILE".
   */
  logging_resource_adaptor(Upstream* upstream, std::string const& filename =
                                                   std::getenv("RMM_LOG_FILE"))
      : upstream_{upstream},
        logger_{std::make_shared<spdlog::logger>(
            "RMM", std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                       filename, true /*truncate file*/))} {
    RMM_EXPECTS(nullptr != upstream,
                "Unexpected null upstream resource pointer.");

    auto const csv_header{"Time,Action,Pointer,Size,Stream"};
    logger_->set_pattern("%v");
    logger_->info(csv_header);
    logger_->set_pattern("%H:%M:%S:%f,%v");
  }

  /**
   * @brief Return pointer to the upstream resource.
   *
   * @return Upstream* Pointer to the upstream resource.
   */
  Upstream* get_upstream() const noexcept { return upstream_; }

  /**
   * @brief Checks whether the upstream resource supports streams.
   *
   * @return true The upstream resource supports streams
   * @return false The upstream resource does not support streams.
   */
  bool supports_streams() const noexcept override {
    return upstream_->supports_streams();
  }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   * 
   * @return bool true if the upstream resource supports get_mem_info, false otherwise.
   */
  bool supports_get_mem_info() const noexcept override { return upstream_->supports_streams(); }

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using the upstream
   * resource and logs the allocation.
   *
   * If the upstream allocation is successful logs the
   * following CSV formatted line to the file specified at construction:
   * ```
   * *TIMESTAMP*,"allocate",*bytes*,*stream*
   * ```
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   * by the upstream resource.
   *
   * @param bytes The size, in bytes, of the allocation
   * @param stream Stream on which to perform the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, cudaStream_t stream) override {
    auto const p = upstream_->allocate(bytes, stream);
    std::string msg{"allocate,"};
    std::stringstream ss;
    ss << p;
    msg += ss.str();
    msg += ",";
    msg += std::to_string(bytes);
    msg += ",";
    msg += std::to_string(reinterpret_cast<uintptr_t>(stream));
    logger_->info(msg);
    return p;
  }

  /**
   * @brief Free allocation of size `bytes` pointed to by `p` and log the
   * deallocation.
   *
   * Every invocation of `logging_resource_adaptor::do_deallocate` will write
   * the following CSV formatted line to the file specified at construction:
   * ```
   * *TIMESTAMP*,"free",*bytes*,*stream*
   * ```
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   * @param bytes Size of the allocation
   * @param stream Stream on which to perform the deallocation
   */
  void do_deallocate(void* p, std::size_t bytes, cudaStream_t stream) override {
    std::string msg{"free,"};
    std::stringstream ss;
    ss << p;
    msg += ss.str();
    msg += ",";
    msg += std::to_string(bytes);
    msg += ",";
    msg += std::to_string(reinterpret_cast<uintptr_t>(stream));
    logger_->info(msg);
    upstream_->deallocate(p, bytes, stream);
  }

  /**
   * @brief Compare the upstream resource to another.
   *
   * @throws Nothing.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  bool do_is_equal(device_memory_resource const &other) const noexcept override {
    if (this == &other)
      return true;
    else {
      logging_resource_adaptor<Upstream> const *cast =
          dynamic_cast<logging_resource_adaptor<Upstream> const *>(&other);
      if (cast != nullptr)
        return upstream_->is_equal(*cast->get_upstream());
      else
        return upstream_->is_equal(other);
    }
  }

  /**
   * @brief Get free and available memory from upstream resource.
   *
   * @throws `rmm::cuda_error` if unable to retrieve memory info.
   *
   * @param stream Stream on which to get the mem info.
   * @return std::pair contaiing free_size and total_size of memory
   */
  std::pair<size_t, size_t> do_get_mem_info(cudaStream_t stream) const override {
    return upstream_->get_mem_info(stream);
  }

  std::shared_ptr<spdlog::logger> logger_;  ///< spdlog logger object

  Upstream* upstream_;  ///< The upstream resource used for satisfying
                        ///< allocation requests
};

/**
 * @brief Convenience factory to return a `logging_resource_adaptor` around the
 * upstream resource `upstream`.
 *
 * @tparam Upstream Type of the upstream `device_memory_resource`.
 * @param upstream Pointer to the upstream resource
 * @param filename Name of the file to write log info. If not specified,
 * retrieves the log file name from the environment variable "RMM_LOG_FILE".
 */
template <typename Upstream>
logging_resource_adaptor<Upstream> make_logging_adaptor(
    Upstream* upstream,
    std::string const& filename = std::getenv("RMM_LOG_FILE")) {
  return logging_resource_adaptor<Upstream>{upstream, filename};
}

}  // namespace mr
}  // namespace rmm
