#pragma once

#include <cuda.h>

#include <Eigen/Dense>

namespace cupoch {
namespace geometry {

namespace distance_test {
__host__ __device__ inline float PointLine(const Eigen::Vector3f &p,
                                           const Eigen::Vector3f &q1,
                                           const Eigen::Vector3f &q2);

__host__ __device__ inline float PointPlane(const Eigen::Vector3f &p,
                                            const Eigen::Vector3f &vert0,
                                            const Eigen::Vector3f &vert1,
                                            const Eigen::Vector3f &vert2);

__host__ __device__ inline float PointAABBSquared(
        const Eigen::Vector3f &p,
        const Eigen::Vector3f &min_bound,
        const Eigen::Vector3f &max_bound);

__host__ __device__ inline float LineSegmentLineSegmentSquared(
        const Eigen::Vector3f &p0,
        const Eigen::Vector3f &q0,
        const Eigen::Vector3f &p1,
        const Eigen::Vector3f &q1,
        float &param0,
        float &param1,
        Eigen::Vector3f &c0,
        Eigen::Vector3f &c1);

}  // namespace distance_test

}  // namespace geometry
}  // namespace cupoch

#include "cupoch/geometry/distance_test.inl"