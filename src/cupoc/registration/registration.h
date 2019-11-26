#pragma once
#include "cupoc/utility/eigen.h"
#include "cupoc/registration/transformation_estimation.h"
#include <thrust/host_vector.h>

namespace cupoc {

namespace geometry {
class PointCloud;
}

namespace registration {

class ICPConvergenceCriteria {
public:
    ICPConvergenceCriteria(float relative_fitness = 1e-6,
                           float relative_rmse = 1e-6,
                           int max_iteration = 30)
        : relative_fitness_(relative_fitness),
          relative_rmse_(relative_rmse),
          max_iteration_(max_iteration) {}
    ~ICPConvergenceCriteria() {}

public:
    float relative_fitness_;
    float relative_rmse_;
    int max_iteration_;
};


class RegistrationResult {
public:
    RegistrationResult(
            const Eigen::Matrix4f &transformation = Eigen::Matrix4f::Identity());
    RegistrationResult(const RegistrationResult& other);
    ~RegistrationResult();

    void SetCorrespondenceSet(const thrust::host_vector<Eigen::Vector2i>& corres);
    thrust::host_vector<Eigen::Vector2i> GetCorrespondenceSet() const;

public:
    Eigen::Matrix4f_u transformation_;
    CorrespondenceSet correspondence_set_;
    float inlier_rmse_;
    float fitness_;
};

/// Functions for ICP registration
RegistrationResult RegistrationICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        float max_correspondence_distance,
        const Eigen::Matrix4f &init = Eigen::Matrix4f::Identity(),
        const TransformationEstimation &estimation =
                TransformationEstimationPointToPoint(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

}
}