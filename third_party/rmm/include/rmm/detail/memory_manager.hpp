/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

/** ---------------------------------------------------------------------------*
 * @brief Memory Manager class
 * 
 * Note: assumes at least C++11
 * --------------------------------------------------------------------------**/

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <vector>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <set>
#include <mutex>

#include <rmm/mr/device/device_memory_resource.hpp>

#include <rmm/rmm_api.h>
#include <cnmem.h>

/** ---------------------------------------------------------------------------*
 * @brief Macro wrapper for CNMEM API calls to return appropriate RMM errors.
 * --------------------------------------------------------------------------**/
#define RMM_CHECK_CNMEM(call) do {            \
  cnmemStatus_t error = (call);             \
  switch (error) {                          \
  case CNMEM_STATUS_SUCCESS:                \
      break; /* don't return on success! */ \
  case CNMEM_STATUS_CUDA_ERROR:             \
      return RMM_ERROR_CUDA_ERROR;          \
  case CNMEM_STATUS_INVALID_ARGUMENT:       \
      return RMM_ERROR_INVALID_ARGUMENT;    \
  case CNMEM_STATUS_NOT_INITIALIZED:        \
      return RMM_ERROR_NOT_INITIALIZED;     \
  case CNMEM_STATUS_OUT_OF_MEMORY:          \
      return RMM_ERROR_OUT_OF_MEMORY;       \
  case CNMEM_STATUS_UNKNOWN_ERROR:          \
  default:                                  \
      return RMM_ERROR_UNKNOWN;             \
  }                                         \
} while(0)

// forward declaration
typedef struct CUstream_st *cudaStream_t;

namespace rmm 
{
  /** -------------------------------------------------------------------------*
   * @brief An event logger for RMM
   * 
   * Calling record() records various data about a memory manager event,
   * includingthe type of event (alloc, free, realloc), the start and end time,
   * the device, the pointer, free and total available memory, the size (for
   * alloc/realloc), the stream, and location (file and line).
   * 
   * The log can be retreived as a CSV stream using to_csv().
   * 
   * ------------------------------------------------------------------------**/
  class Logger
  {
  public:        
      Logger() { base_time = std::chrono::system_clock::now(); }

      enum MemEvent_t {
          Alloc = 0,
          Realloc,
          Free
      };

      using TimePt = std::chrono::system_clock::time_point;

      /// Record a memory manager event in the log.
      void record(MemEvent_t event, int deviceId, void* ptr,
                  TimePt start, TimePt end, 
                  size_t freeMem, size_t totalMem,
                  size_t size, cudaStream_t stream,
                  std::string filename,
                  unsigned int line);

      /// Clear the log
      void clear();
      
      /// Write the log to comma-separated value file
      void to_csv(std::ostream &csv);
  private:
      std::set<void*> current_allocations;

      struct MemoryEvent {
          MemEvent_t event;
          int deviceId;
          void* ptr;
          size_t size;
          cudaStream_t stream;
          size_t freeMem;
          size_t totalMem;
          size_t currentAllocations;
          TimePt start;
          TimePt end;
          std::string filename;
          unsigned int line;
      };
      
      TimePt base_time;
      std::vector<MemoryEvent> events;
      std::mutex log_mutex;
  };

  /** -------------------------------------------------------------------------*
   * @brief RMM Manager class maintains the memory manager context, including
   * the RMM event log, configuration options, and registered streams.
   * 
   * Manager is a singleton class, and should be accessed via getInstance(). 
   * A number of static convenience methods are provided that wrap getInstance(),
   * such as getLogger() and getOptions().
   * ------------------------------------------------------------------------**/
  class Manager
  {
  public:
    /** -----------------------------------------------------------------------*
     * @brief Get the Manager instance singleton object
     * 
     * @return Manager& the Manager singleton
     * ----------------------------------------------------------------------**/
    static Manager& getInstance(){
        // Myers' singleton. Thread safe and unique. Note: C++11 required.
        static Manager instance;
        return instance;
    }

    /** -----------------------------------------------------------------------*
     * @brief Get the RMM Logger object
     * 
     * @return Logger& the logger object
     * ----------------------------------------------------------------------**/
    static Logger& getLogger() { return getInstance().logger; }

    /** -----------------------------------------------------------------------*
     * @brief Initialize RMM options
     * 
     * Accepts an optional rmmOptions_t struct that describes the settings used
     * to initialize the memory manager. If no `options` is passed, default
     * options are used.
     * 
     * @param[in] options Optional options to set
     * ----------------------------------------------------------------------**/
    void initialize(const rmmOptions_t *options = nullptr);

    /** -----------------------------------------------------------------------*
     * @brief Shut down the Manager (clears the context)
     * 
     * ----------------------------------------------------------------------**/
    void finalize();

    /** -----------------------------------------------------------------------*
     * @brief Check whether the Manager has been initialized.
     * 
     * @return true if Manager has been initialized.
     * @return false if Manager has not been initialized.
     * ----------------------------------------------------------------------**/
    bool isInitialized() {
        return getInstance().is_initialized;
    }

    /** -----------------------------------------------------------------------*
     * @brief Get the Options object
     * 
     * @return rmmOptions_t the currently set RMM options
     * ----------------------------------------------------------------------**/
    static rmmOptions_t getOptions() { return getInstance().options; }

    /** -----------------------------------------------------------------------*
     * @brief Returns true when pool allocation is enabled
     * 
     * @return true if pool allocation is enabled
     * @return false if pool allocation is disabled
     * ----------------------------------------------------------------------**/
    static inline bool usePoolAllocator() {
        return getOptions().allocation_mode & PoolAllocation;
    }

    /** -----------------------------------------------------------------------*
     * @brief Returns true if CUDA Managed Memory allocation is enabled
     * 
     * @return true if CUDA Managed Memory allocation is enabled
     * @return false if CUDA Managed Memory allocation is disabled
     * ----------------------------------------------------------------------**/
    static inline bool useManagedMemory() {
        return getOptions().allocation_mode & CudaManagedMemory;
    }

    /** -----------------------------------------------------------------------*
     * @brief Returns true when CUDA default allocation is enabled
     *          * 
     * @return true if CUDA default allocation is enabled
     * @return false if CUDA default allocation is disabled
     * ----------------------------------------------------------------------**/
    inline bool useCudaDefaultAllocator() {
        return CudaDefaultAllocation == getOptions().allocation_mode;
    }

    /** -----------------------------------------------------------------------*
     * @brief Register a new stream into the device memory manager.
     * 
     * Also returns success if the stream is already registered.
     * 
     * @param stream The stream to register
     * @return rmmError_t RMM_SUCCESS if all goes well,
     *                    RMM_ERROR_INVALID_ARGUMENT if the stream is invalid.
     * ----------------------------------------------------------------------**/
    rmmError_t registerStream(cudaStream_t stream);

  private:
    Manager() = default;
    ~Manager() = default;
    Manager(const Manager&) = delete;
    Manager& operator=(const Manager&) = delete;
    std::mutex manager_mutex;
    std::set<cudaStream_t> registered_streams;
    Logger logger;

    rmmOptions_t options{};
    bool is_initialized{false};

    std::unique_ptr<rmm::mr::device_memory_resource> initialized_resource{};
  };
}

#endif // MEMORY_MANAGER_H
