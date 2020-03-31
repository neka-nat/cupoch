#include "cupoch/geometry/distance_test.h"

namespace cupoch {
namespace geometry {

namespace distance_test {

float PointLine(const Eigen::Vector3f &p,
                const Eigen::Vector3f &q1,
                const Eigen::Vector3f &q2) {
    const float d_q12 = (q1 - q2).norm();
    if (d_q12 == 0) return (p - q1).norm();
    return (p - q1).cross(p - q2).norm() / d_q12;
}

float PointPlane(const Eigen::Vector3f &p,
                 const Eigen::Vector3f &vert0,
                 const Eigen::Vector3f &vert1,
                 const Eigen::Vector3f &vert2) {
    auto n = (vert1 - vert0).cross(vert2 - vert0);
    const float dn = n.norm();
    if (dn == 0) return PointLine(p, vert0, vert1);
    n /= dn;
    return abs(n.dot(p - vert0));
}

}  // namespace distance_test

}  // namespace geometry
}  // namespace cupoch