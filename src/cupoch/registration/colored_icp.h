#pragma once

#include <Eigen/Core>

#include "cupoch/registration/registration.h"

namespace cupoch {

namespace geometry {
class PointCloud;
}

namespace registration {
class RegistrationResult;

/// Function to align colored point clouds
/// This is implementation of following paper
/// J. Park, Q.-Y. Zhou, V. Koltun,
/// Colored Point Cloud Registration Revisited, ICCV 2017
RegistrationResult RegistrationColoredICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        float max_distance,
        const Eigen::Matrix4f &init = Eigen::Matrix4f::Identity(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria(),
        float lambda_geometric = 0.968);

}  // namespace registration
}  // namespace cupoch