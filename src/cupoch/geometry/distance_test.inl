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

float PointAABBSquared(const Eigen::Vector3f &p,
                       const Eigen::Vector3f &min_bound,
                       const Eigen::Vector3f &max_bound) {
    float dist2 = 0.0f;
    for (int i = 0; i < 3; i++) {
        float v = p[i];
        if (v < min_bound[i]) dist2 += (min_bound[i] - v) * (min_bound[i] - v);
        if (v > max_bound[i]) dist2 += (v - max_bound[i]) * (v - max_bound[i]);
    }
    return dist2;
}

float LineSegmentLineSegmentSquared(const Eigen::Vector3f &p0,
                                    const Eigen::Vector3f &q0,
                                    const Eigen::Vector3f &p1,
                                    const Eigen::Vector3f &q1,
                                    float &param0,
                                    float &param1,
                                    Eigen::Vector3f &c0,
                                    Eigen::Vector3f &c1) {
    const Eigen::Vector3f d0 = q0 - p0;
    const Eigen::Vector3f d1 = q1 - p1;
    const Eigen::Vector3f r = p0 - p1;
    const float a = d0.squaredNorm();
    const float e = d1.squaredNorm();
    const float f = d1.dot(r);
    if (a <= std::numeric_limits<float>::epsilon() &&
        e <= std::numeric_limits<float>::epsilon()) {
        param0 = 0.0;
        param1 = 0.0;
        c0 = p0;
        c1 = p1;
        return (c0 - c1).squaredNorm();
    }
    if (a <= std::numeric_limits<float>::epsilon()) {
        param0 = 0.0;
        param1 = min(max(f / e, 0.0), 1.0);
    } else {
        float c = d0.dot(r);
        if (e <= std::numeric_limits<float>::epsilon()) {
            param1 = 0.0;
            param0 = min(max(-c / a, 0.0), 1.0);
        } else {
            const float b = d0.dot(d1);
            const float denom = a * e - b * b;
            if (denom != 0.0) {
                param0 = min(max((b * f - c * e) / denom, 0.0), 1.0);
            } else {
                param0 = 0.0;
            }
            param1 = (b * param0 + f) / e;
            if (param1 < 0.0) {
                param1 = 0.0;
                param0 = min(max(-c / a, 0.0), 1.0);
            } else if (param1 > 1.0) {
                param1 = 1.0;
                param0 = min(max((b - c) / a, 0.0), 1.0);
            }
        }
    }
    c0 = p0 + param0 * d0;
    c1 = p1 + param1 * d1;
    return (c0 - c1).squaredNorm();
}

}  // namespace distance_test

}  // namespace geometry
}  // namespace cupoch