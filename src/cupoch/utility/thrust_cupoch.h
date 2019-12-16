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
#else
template<typename T>
using device_vector = thrust::device_vector<T>;
#endif

}
}