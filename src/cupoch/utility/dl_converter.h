#pragma once

#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <dlpack/dlpack.h>

namespace cupoch {
namespace utility {

template<typename T, int Dim>
DLManagedTensor* ToDLPack(const thrust::device_vector<Eigen::Matrix<T, Dim, 1>>& src);

template<typename T, int Dim>
thrust::device_vector<Eigen::Matrix<T, Dim, 1>> FromDLPack(const DLManagedTensor* src);

}
}