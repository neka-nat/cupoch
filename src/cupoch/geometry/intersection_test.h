#pragma once

#include <Eigen/Dense>
#include <cuda.h>

namespace cupoch {
namespace geometry {

namespace intersection_test {
    __host__ __device__
    inline bool TriangleTriangle3d(const Eigen::Vector3f& p0,
                                   const Eigen::Vector3f& p1,
                                   const Eigen::Vector3f& p2,
                                   const Eigen::Vector3f& q0,
                                   const Eigen::Vector3f& q1,
                                   const Eigen::Vector3f& q2);

    __host__ __device__
    inline bool TriangleAABB(const Eigen::Vector3f& box_center,
                             const Eigen::Vector3f& box_half_size,
                             const Eigen::Vector3f& vert0,
                             const Eigen::Vector3f& vert1,
                             const Eigen::Vector3f& vert2);
}

}
}

#include "cupoch/geometry/intersection_test.inl"