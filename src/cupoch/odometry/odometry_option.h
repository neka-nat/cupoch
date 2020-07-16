/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
**/
#pragma once

#include <vector>

#include "cupoch/utility/eigen.h"

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