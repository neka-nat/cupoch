#pragma once
#include <cupoch/utility/eigen.h>
#include <thrust/device_vector.h>

namespace cupoch {
namespace geometry {

Eigen::Vector3f_u ComputeMinBound(const thrust::device_vector<Eigen::Vector3f_u>& points);

Eigen::Vector3f_u ComputeMaxBound(const thrust::device_vector<Eigen::Vector3f_u>& points);

Eigen::Vector3f_u ComuteCenter(const thrust::device_vector<Eigen::Vector3f_u>& points);

void TransformPoints(const Eigen::Matrix4f& transformation,
                     thrust::device_vector<Eigen::Vector3f_u>& points);

void TransformNormals(const Eigen::Matrix4f& transformation,
                      thrust::device_vector<Eigen::Vector3f_u>& normals);

}
}