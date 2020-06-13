#pragma once
#include <thrust/host_vector.h>

#include "cupoch/registration/transformation_estimation.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {

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
    RegistrationResult(const Eigen::Matrix4f &transformation =
                               Eigen::Matrix4f::Identity());
    RegistrationResult(const RegistrationResult &other);
    ~RegistrationResult();

    void SetCorrespondenceSet(
            const thrust::host_vector<Eigen::Vector2i> &corres);
    thrust::host_vector<Eigen::Vector2i> GetCorrespondenceSet() const;

public:
    Eigen::Matrix4f_u transformation_;
    CorrespondenceSet correspondence_set_;
    float inlier_rmse_;
    float fitness_;
};

/// \brief Function for evaluating registration between point clouds.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_correspondence_distance Maximum correspondence points-pair
/// distance. \param transformation The 4x4 transformation matrix to transform
/// source to target. Default value: array([[1., 0., 0., 0.], [0., 1., 0., 0.],
/// [0., 0., 1., 0.], [0., 0., 0., 1.]]).
RegistrationResult EvaluateRegistration(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        float max_correspondence_distance,
        const Eigen::Matrix4f &transformation = Eigen::Matrix4f::Identity());

/// Functions for ICP registration
RegistrationResult RegistrationICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        float max_correspondence_distance,
        const Eigen::Matrix4f &init = Eigen::Matrix4f::Identity(),
        const TransformationEstimation &estimation =
                TransformationEstimationPointToPoint(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

}  // namespace registration
}  // namespace cupoch