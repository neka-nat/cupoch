#pragma once

#include "cupoc/utility/eigen.h"
#include <thrust/device_vector.h>

namespace cupoc {
namespace registration {

Eigen::Matrix4f_u Kabsch(const thrust::device_vector<Eigen::Vector3f_u>& model,
                         const thrust::device_vector<Eigen::Vector3f_u>& target);

}
}