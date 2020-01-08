#pragma once

#include <Eigen/Core>
#include <thrust/device_vector.h>

namespace dlpack {
class DLTContainer;
}

namespace cupoch {
namespace utility {

void ToDLPack(const thrust::device_vector<Eigen::Vector3f>& src, dlpack::DLTContainer& dst);
void FromDLPack(const dlpack::DLTContainer& src, const thrust::device_vector<Eigen::Vector3f>& dst);

}
}