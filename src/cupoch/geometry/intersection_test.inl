#include <tomasakeninemoeller/opttritri.h>
#include <tomasakeninemoeller/tribox3.h>

#include "cupoch/geometry/intersection_test.h"
#include "cupoch/geometry/distance_test.h"

namespace cupoch {
namespace geometry {

namespace intersection_test {

bool TriangleTriangle3d(const Eigen::Vector3f &p0,
                        const Eigen::Vector3f &p1,
                        const Eigen::Vector3f &p2,
                        const Eigen::Vector3f &q0,
                        const Eigen::Vector3f &q1,
                        const Eigen::Vector3f &q2) {
    const Eigen::Vector3f mu = (p0 + p1 + p2 + q0 + q1 + q2) / 6;
    const Eigen::Vector3f sigma =
            (((p0 - mu).array().square() + (p1 - mu).array().square() +
              (p2 - mu).array().square() + (q0 - mu).array().square() +
              (q1 - mu).array().square() + (q2 - mu).array().square()) /
             5)
                    .sqrt() +
            1e-12;
    Eigen::Vector3f p0m = (p0 - mu).array() / sigma.array();
    Eigen::Vector3f p1m = (p1 - mu).array() / sigma.array();
    Eigen::Vector3f p2m = (p2 - mu).array() / sigma.array();
    Eigen::Vector3f q0m = (q0 - mu).array() / sigma.array();
    Eigen::Vector3f q1m = (q1 - mu).array() / sigma.array();
    Eigen::Vector3f q2m = (q2 - mu).array() / sigma.array();
    return NoDivTriTriIsect(p0m.data(), p1m.data(), p2m.data(), q0m.data(),
                            q1m.data(), q2m.data()) != 0;
}

bool TriangleAABB(const Eigen::Vector3f &box_center,
                  const Eigen::Vector3f &box_half_size,
                  const Eigen::Vector3f &vert0,
                  const Eigen::Vector3f &vert1,
                  const Eigen::Vector3f &vert2) {
    float *tri_verts[3] = {const_cast<float *>(vert0.data()),
                           const_cast<float *>(vert1.data()),
                           const_cast<float *>(vert2.data())};
    return triBoxOverlap(const_cast<float *>(box_center.data()),
                         const_cast<float *>(box_half_size.data()),
                         tri_verts) != 0;
}

bool AABBAABB(const Eigen::Vector3f &min_bound0,
              const Eigen::Vector3f &max_bound0,
              const Eigen::Vector3f &min_bound1,
              const Eigen::Vector3f &max_bound1) {
    return (min_bound0[0] <= max_bound1[0] && max_bound0[0] >= min_bound1[0]) &&
           (min_bound0[1] <= max_bound1[1] && max_bound0[1] >= min_bound1[1]) &&
           (min_bound0[2] <= max_bound1[2] && max_bound0[2] >= min_bound1[2]);
}

bool LineSegmentAABB(const Eigen::Vector3f &p0,
                     const Eigen::Vector3f &p1,
                     const Eigen::Vector3f &min_bound,
                     const Eigen::Vector3f &max_bound) {
    const Eigen::Vector3f center = (min_bound + max_bound) * 0.5;
    const Eigen::Vector3f ext = max_bound - center;
    Eigen::Vector3f mid = (p0 + p1) * 0.5;
    const Eigen::Vector3f dst = p1 - mid;
    mid -= center;
    Eigen::Vector3f absdst;
    for (int i = 0; i < 3; ++i) {
        absdst[i] = abs(dst[i]);
        if (abs(mid[i]) > ext[i] + absdst[i]) return false;
    }
    absdst.array() += std::numeric_limits<float>::epsilon();
    if (abs(mid[1] * dst[2] - mid[2] * dst[1]) > ext[1] * absdst[2] + ext[2] * absdst[1]) return false;
    if (abs(mid[2] * dst[0] - mid[0] * dst[2]) > ext[2] * absdst[0] + ext[0] * absdst[2]) return false;
    if (abs(mid[0] * dst[1] - mid[1] * dst[0]) > ext[0] * absdst[1] + ext[1] * absdst[0]) return false;
    return true;
}

bool SphereAABB(const Eigen::Vector3f& center,
                float radius,
                const Eigen::Vector3f& min_bound,
                const Eigen::Vector3f& max_bound) {
    float dist2 = distance_test::PointAABBSquared(center, min_bound, max_bound);
    return dist2 <= radius * radius;
}

bool BoxBox(const Eigen::Vector3f& extents1,
            const Eigen::Matrix3f& rot1,
            const Eigen::Vector3f& center1,
            const Eigen::Vector3f& extents2,
            const Eigen::Matrix3f& rot2,
            const Eigen::Vector3f& center2) {
    const Eigen::Matrix3f r = rot1 * rot2;
    const Eigen::Vector3f t = rot1 * (center2 - center1);
    const Eigen::Matrix3f absr = r.array().abs() + std::numeric_limits<float>::epsilon();
    for (int i = 0; i < 3; ++i) {
        if (std::abs(t[i]) > extents1[i] + absr.row(i).dot(extents2)) return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (std::abs(t.dot(r.col(i))) > absr.col(i).dot(extents1) + extents2[i]) return false;
    }
    float ra, rb;
    ra = extents1[1] * absr(2, 0) + extents1[2] * absr(1, 0);
    rb = extents2[1] * absr(0, 2) + extents2[2] * absr(0, 1);
    if (std::abs(t[2] * r(1, 0) - t[1] * r(2, 0)) > ra + rb) return false;

    ra = extents1[1] * absr(2, 1) + extents1[2] * absr(1, 1);
    rb = extents2[0] * absr(0, 2) + extents2[2] * absr(0, 0);
    if (std::abs(t[2] * r(1, 1) - t[1] * r(2, 1)) > ra + rb) return false;

    ra = extents1[1] * absr(2, 2) + extents1[2] * absr(1, 2);
    rb = extents2[0] * absr(0, 1) + extents2[1] * absr(0, 0);
    if (std::abs(t[2] * r(1, 2) - t[1] * r(2, 2)) > ra + rb) return false;

    ra = extents1[0] * absr(2, 0) + extents1[2] * absr(0, 0);
    rb = extents2[1] * absr(1, 2) + extents2[2] * absr(1, 1);
    if (std::abs(t[0] * r(2, 0) - t[2] * r(0, 0)) > ra + rb) return false;

    ra = extents1[0] * absr(2, 1) + extents1[2] * absr(0, 1);
    rb = extents2[0] * absr(1, 2) + extents2[2] * absr(1, 0);
    if (std::abs(t[0] * r(2, 1) - t[2] * r(0, 1)) > ra + rb) return false;

    ra = extents1[0] * absr(2, 2) + extents1[2] * absr(0, 2);
    rb = extents2[0] * absr(1, 1) + extents2[1] * absr(1, 0);
    if (std::abs(t[0] * r(2, 2) - t[2] * r(0, 2)) > ra + rb) return false;

    ra = extents1[0] * absr(1, 0) + extents1[1] * absr(0, 0);
    rb = extents2[1] * absr(2, 2) + extents2[2] * absr(2, 1);
    if (std::abs(t[1] * r(0, 1) - t[0] * r(1, 0)) > ra + rb) return false;

    ra = extents1[0] * absr(1, 1) + extents1[1] * absr(0, 1);
    rb = extents2[0] * absr(2, 2) + extents2[2] * absr(2, 0);
    if (std::abs(t[1] * r(0, 1) - t[0] * r(1, 1)) > ra + rb) return false;

    ra = extents1[0] * absr(1, 2) + extents1[1] * absr(0, 2);
    rb = extents2[0] * absr(2, 1) + extents2[1] * absr(2, 0);
    if (std::abs(t[1] * r(0, 2) - t[0] * r(1, 2)) > ra + rb) return false;

    return true;
}

}  // namespace intersection_test

}  // namespace geometry
}  // namespace cupoch