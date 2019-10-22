#pragma once
#include <cupoc/utility/eigen.h>
#include <thrust/device_vector.h>

namespace cupoc {
namespace geometry {

Eigen::Vector3f_u ComputeMinBound(const thrust::device_vector<Eigen::Vector3f_u>& points);

Eigen::Vector3f_u ComputeMaxBound(const thrust::device_vector<Eigen::Vector3f_u>& points);

Eigen::Vector3f_u ComuteCenter(const thrust::device_vector<Eigen::Vector3f_u>& points);

}
}