#pragma once

#include <tuple>

#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "cupoch/odometry/odometry_option.h"
#include "cupoch/odometry/rgbdodometry_jacobian.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {

namespace geometry {
class RGBDImage;
}

namespace odometry {
/// Function to estimate 6D odometry between two RGB-D images
/// output: is_success, 4x4 motion matrix, 6x6 information matrix
std::tuple<bool, Eigen::Matrix4f, Eigen::Matrix6f> ComputeRGBDOdometry(
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic =
                camera::PinholeCameraIntrinsic(),
        const Eigen::Matrix4f &odo_init = Eigen::Matrix4f::Identity(),
        const RGBDOdometryJacobian &jacobian_method =
                RGBDOdometryJacobianFromHybridTerm(),
        const OdometryOption &option = OdometryOption());

}  // namespace odometry
}  // namespace cupoch