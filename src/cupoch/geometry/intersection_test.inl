#include "cupoch/geometry/intersection_test.h"

#include <tomasakeninemoeller/opttritri.h>
#include <tomasakeninemoeller/tribox3.h>

namespace cupoch {
namespace geometry {

namespace intersection_test {

bool TriangleTriangle3d(const Eigen::Vector3f& p0,
                        const Eigen::Vector3f& p1,
                        const Eigen::Vector3f& p2,
                        const Eigen::Vector3f& q0,
                        const Eigen::Vector3f& q1,
                        const Eigen::Vector3f& q2) {
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

}

}
}