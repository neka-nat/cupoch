#pragma once

#if defined(__CUDA_BACKEND__)
#include <thrust/device_vector.h>
#elif defined(__OPENCL_BACKEND__)
#include <boost/compute/container/vector.hpp>
#else
#error "Unsupported backend."
#endif

namespace cupoch {
namespace backend {

#if defined(__CUDA_BACKEND__)
template <class T>
using device_vector = thrust::device_vector<T>;

#elif defined(__OPENCL_BACKEND__)
template <class T, class Alloc = boost::compute::buffer_allocator<T> >
using device_vector = boost::compute::vector<T>;

#else
#error "Unsupported backend."
#endif

}
}