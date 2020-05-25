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

__host__ __device__ inline bool LineSegmentAABB(
        const Eigen::Vector3f &p0,
        const Eigen::Vector3f &p1,
        const Eigen::Vector3f &min_bound,
        const Eigen::Vector3f &max_bound);

__host__ __device__ inline bool SphereAABB(
        const Eigen::Vector3f& center,
        float radius,
        const Eigen::Vector3f& min_bound,
        const Eigen::Vector3f& max_bound);

__host__ __device__ inline bool BoxBox(
        const Eigen::Vector3f& extents1,
        const Eigen::Matrix3f& rot1,
        const Eigen::Vector3f& center1,
        const Eigen::Vector3f& extents2,
        const Eigen::Matrix3f& rot2,
        const Eigen::Vector3f& center2);

}  // namespace intersection_test

}  // namespace geometry
}  // namespace cupoch

#include "cupoch/geometry/intersection_test.inl"