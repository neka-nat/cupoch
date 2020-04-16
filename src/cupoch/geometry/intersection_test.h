#pragma once

#include <cuda.h>

#include <Eigen/Dense>

namespace cupoch {
namespace geometry {

namespace intersection_test {
__host__ __device__ inline bool TriangleTriangle3d(const Eigen::Vector3f &p0,
                                                   const Eigen::Vector3f &p1,
                                                   const Eigen::Vector3f &p2,
                                                   const Eigen::Vector3f &q0,
                                                   const Eigen::Vector3f &q1,
                                                   const Eigen::Vector3f &q2);

__host__ __device__ inline bool TriangleAABB(
        const Eigen::Vector3f &box_center,
        const Eigen::Vector3f &box_half_size,
        const Eigen::Vector3f &vert0,
        const Eigen::Vector3f &vert1,
        const Eigen::Vector3f &vert2);

__host__ __device__ inline bool AABBAABB(
        const Eigen::Vector3f &min_bound0,
        const Eigen::Vector3f &max_bound0,
        const Eigen::Vector3f &min_bound1,
        const Eigen::Vector3f &max_bound1);

__host__ __device__ inline bool SphereAABB(
        const Eigen::Vector3f& center,
        float radius,
        const Eigen::Vector3f& min_bound,
        const Eigen::Vector3f& max_bound);

}  // namespace intersection_test

}  // namespace geometry
}  // namespace cupoch

#include "cupoch/geometry/intersection_test.inl"