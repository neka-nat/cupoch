#pragma once

#include <dlpack/dlpack.h>
#include <Eigen/Core>
#include <thrust/device_vector.h>

namespace cupoch {
namespace utility {

DLManagedTensor* ToDLPack(const thrust::device_vector<Eigen::Vector3f>& src);
thrust::device_vector<Eigen::Vector3f> FromDLPack(const DLManagedTensor* src);

}
}