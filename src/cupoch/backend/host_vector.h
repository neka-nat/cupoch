#pragma once

#if defined(__CUDA_BACKEND__)
#include <thrust/host_vector.h>
#elif defined(__OPENCL_BACKEND__)
#include <vector>
#else
#error "Unsupported backend."
#endif

namespace cupoch {
namespace backend {

#if defined(__CUDA_BACKEND__)
template <class T>
using host_vector = thrust::host_vector<T>;

#elif defined(__OPENCL_BACKEND__)
template <class T>
using host_vector = std::vector<T>;

#else
#error "Unsupported backend."
#endif

}
}