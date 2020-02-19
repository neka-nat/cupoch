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
#define exec_policy_on(stream) (rmm::exec_policy(stream)->on(stream))
inline void InitializeCupoch() {
    rmmOptions_t options{static_cast<rmmAllocationMode_t>(PoolAllocation | CudaManagedMemory), 0, true};
    rmmInitialize(&options);
}

#else
template<typename T>
using device_vector = thrust::device_vector<T>;
#define exec_policy_on(stream) (thrust::cuda::par.on(stream))
inline void InitializeCupoch() {}

#endif

}
}