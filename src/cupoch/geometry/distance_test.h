#pragma once

#include <cuda.h>

#include <Eigen/Dense>

namespace cupoch {
namespace geometry {

namespace distance_test {
__host__ __device__ inline float PointLine(const Eigen::Vector3f &p,
                                           const Eigen::Vector3f &q1,
                                           const Eigen::Vector3f &q2);

__host__ __device__ inline float PointPlane(
        const Eigen::Vector3f &p,
        const Eigen::Vector3f &vert0,
        const Eigen::Vector3f &vert1,
        const Eigen::Vector3f &vert2);

}  // namespace distance_test

}  // namespace geometry
}  // namespace cupoch

#include "cupoch/geometry/distance_test.inl"