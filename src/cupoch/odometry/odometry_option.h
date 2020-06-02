#pragma once

#include "cupoch/utility/eigen.h"
#include <vector>

namespace cupoch {
namespace odometry {

class OdometryOption {
public:
    OdometryOption(
            const std::vector<int> &iteration_number_per_pyramid_level =
                    {20, 10,
                     5} /* {smaller image size to original image size} */,
            float max_depth_diff = 0.03,
            float min_depth = 0.0,
            float max_depth = 4.0,
            float nu = 5.0,
            float sigma2_init = 1.0,
            const Eigen::Vector6f &inv_sigma_mat_diag = Eigen::Vector6f::Zero())
        : iteration_number_per_pyramid_level_(
                  iteration_number_per_pyramid_level),
          max_depth_diff_(max_depth_diff),
          min_depth_(min_depth),
          max_depth_(max_depth),
          nu_(nu),
          sigma2_init_(sigma2_init),
          inv_sigma_mat_diag_(inv_sigma_mat_diag) {}
    ~OdometryOption() {}

public:
    std::vector<int> iteration_number_per_pyramid_level_;
    float max_depth_diff_;
    float min_depth_;
    float max_depth_;
    float nu_;
    float sigma2_init_;
    Eigen::Vector6f inv_sigma_mat_diag_;
};

}  // namespace odometry
}  // namespace cupoch