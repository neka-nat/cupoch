#pragma once

#ifdef USE_RMM
#include <rmm/thrust_rmm_allocator.h>
#include <rmm/rmm_api.h>
#else
#include <thrust/device_vector.h>
#endif

namespace cupoch {
namespace utility {

#ifdef USE_RMM
template <typename T>
using device_vector = rmm::device_vector<T>;

inline decltype(auto) exec_policy(cudaStream_t stream = 0) {
    return rmm::exec_policy(stream);
}

inline void InitializeAllocator(
        rmmAllocationMode_t mode = CudaDefaultAllocation,
        size_t initial_pool_size = 0,
        bool logging = false,
        const std::vector<int> &devices = {}) {
    static bool is_initialized = false;
    if (is_initialized) rmmFinalize();
    rmmOptions_t options = {mode, initial_pool_size, logging, devices};
    rmmInitialize(&options);
    is_initialized = true;
}

#else
template <typename T>
using device_vector = thrust::device_vector<T>;

inline decltype(auto) exec_policy(cudaStream_t stream = 0) {
    return &thrust::cuda::par;
}

inline void InitializeAllocator(
        rmmAllocationMode_t mode = CudaDefaultAllocation,
        size_t initial_pool_size = 0,
        bool logging = false,
        const std::vector<int> &devices = {}) {}

#endif

}  // namespace utility
}  // namespace cupoch