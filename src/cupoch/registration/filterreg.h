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
#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace geometry {
class PointCloud;
}

namespace registration {

class FilterRegResult {
public:
    FilterRegResult(
            const Eigen::Matrix4f &transformation = Eigen::Matrix4f::Identity())
        : transformation_(transformation){};
    FilterRegResult(const FilterRegResult &other)
        : transformation_(other.transformation_),
          likelihood_(other.likelihood_){};
    ~FilterRegResult(){};

public:
    Eigen::Matrix4f_u transformation_;
    float likelihood_;
};

class FilterRegOption {
public:
    FilterRegOption(float sigma_initial = 0.1,
                    float sigma_min = 1.0e-4,
                    float relative_likelihood = 1.0e-6,
                    int max_iteration = 30)
        : sigma_initial_(sigma_initial),
          sigma_min_(sigma_min),
          relative_likelihood_(relative_likelihood),
          max_iteration_(max_iteration){};
    ~FilterRegOption(){};

public:
    float sigma_initial_;
    float sigma_min_;
    float relative_likelihood_;
    int max_iteration_;
};

/// Functions for FilterReg registration
FilterRegResult RegistrationFilterReg(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const Eigen::Matrix4f &init = Eigen::Matrix4f::Identity(),
        const FilterRegOption &option = FilterRegOption());

}  // namespace registration
}  // namespace cupoch