#pragma once

#ifdef USE_RMM
#include <rmm/thrust_rmm_allocator.h>
#else
#include <thrust/device_vector.h>
#endif

namespace cupoch {
namespace utility {

#ifdef USE_RMM
template<typename T>
using device_vector = rmm::device_vector<T>;

inline decltype(auto) exec_policy(cudaStream_t stream = 0) {
    return rmm::exec_policy(stream);
}

inline void InitializeCupoch() {
    rmmOptions_t options{static_cast<rmmAllocationMode_t>(PoolAllocation | CudaManagedMemory), 0, true};
    rmmInitialize(&options);
}

#else
template<typename T>
using device_vector = thrust::device_vector<T>;

inline decltype(auto) exec_policy(cudaStream_t stream = 0) {
    return &thrust::cuda::par;
}

inline void InitializeCupoch() {}

#endif

}
}