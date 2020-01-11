#pragma once

#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <dlpack/dlpack.h>

namespace cupoch {
namespace utility {

void ToDLPack(const thrust::device_vector<Eigen::Vector3f>& src, DLManagedTensor** dst);
void FromDLPack(const DLManagedTensor* src, const thrust::device_vector<Eigen::Vector3f>& dst);

}
}