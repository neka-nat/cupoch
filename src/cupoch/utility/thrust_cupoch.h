#pragma once

#ifdef USE_RMM
#include <rmm/thrust_rmm_allocator.h>
#else
#include <thrust/device_vector.h>
#endif

namespace cupoch {
namespace thrustcupoch {

#ifdef USE_RMM
template<typename T>
using device_vector = rmm::device_vector<T>;
#define exec_policy_on(stream) (rmm::exec_policy(stream)->on(stream))
#else
template<typename T>
using device_vector = thrust::device_vector<T>;
#define exec_policy_on(stream) (thrust::cuda::par.on(stream))
#endif

}
}