#pragma once
#include <cupoch/utility/eigen.h>
#include <thrust/device_vector.h>

namespace cupoch {
namespace geometry {

Eigen::Vector3f ComputeMinBound(const thrust::device_vector<Eigen::Vector3f>& points);

Eigen::Vector3f ComputeMaxBound(const thrust::device_vector<Eigen::Vector3f>& points);

Eigen::Vector3f ComuteCenter(const thrust::device_vector<Eigen::Vector3f>& points);

void ResizeAndPaintUniformColor(thrust::device_vector<Eigen::Vector3f>& colors,
                                const size_t size,
                                const Eigen::Vector3f& color);

void TransformPoints(const Eigen::Matrix4f& transformation,
                     thrust::device_vector<Eigen::Vector3f>& points);

void TransformNormals(const Eigen::Matrix4f& transformation,
                      thrust::device_vector<Eigen::Vector3f>& normals);

}
}