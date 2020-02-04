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
}

}
}

#include "cupoch/geometry/intersection_test.inl"