#pragma once

#include <Eigen/Core>
#include <memory>

#include "cupoch/registration/registration.h"
#include "cupoch/registration/transformation_estimation.h"

namespace cupoch {
namespace registration {

class RegistrationResult;

class TransformationEstimationForGeneralizedICP
    : public TransformationEstimation {
public:
    ~TransformationEstimationForGeneralizedICP() override = default;

    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };

    explicit TransformationEstimationForGeneralizedICP(
            float epsilon = 1e-3)
        : epsilon_(epsilon) {}

public:
    float ComputeRMSE(const geometry::PointCloud &source,
                       const geometry::PointCloud &target,
                       const CorrespondenceSet &corres) const override;

    Eigen::Matrix4f ComputeTransformation(
            cudaStream_t stream1, cudaStream_t stream2,
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

public:
    /// Small constant representing covariance along the normal.
    float epsilon_ = 1e-3;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::GeneralizedICP;
};

/// \brief Function for Generalized ICP registration.
///
/// This is implementation of following paper
//  A. Segal, D .Haehnel, S. Thrun
/// Generalized-ICP, RSS 2009.
///
/// \param source The source point cloud.
/// \param target The target point cloud.
/// \param max_distance Maximum correspondence points-pair distance.
/// \param init Initial transformation estimation.
/// Default value: array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
/// [0., 0., 0., 1.]]). \param criteria  Convergence criteria. \param
RegistrationResult RegistrationGeneralizedICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        float max_correspondence_distance,
        const Eigen::Matrix4f &init = Eigen::Matrix4f::Identity(),
        const TransformationEstimationForGeneralizedICP &estimation =
                TransformationEstimationForGeneralizedICP(),
        const ICPConvergenceCriteria &criteria = ICPConvergenceCriteria());

}  // namespace registration
}  // namespace cupoch
