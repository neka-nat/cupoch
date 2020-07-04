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