#pragma once

#include "cupoch/utility/eigen.h"
#include "cupoch/registration/transformation_estimation.h"
#include <thrust/device_vector.h>

namespace cupoch {
namespace registration {

Eigen::Matrix4f_u Kabsch(const thrust::device_vector<Eigen::Vector3f>& model,
                         const thrust::device_vector<Eigen::Vector3f>& target,
                         const CorrespondenceSet& corres);

Eigen::Matrix4f_u Kabsch(const thrust::device_vector<Eigen::Vector3f>& model,
                         const thrust::device_vector<Eigen::Vector3f>& target);

}
}