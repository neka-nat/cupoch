#include "cupoch/registration/generalized_icp.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "cupoch/knn/kdtree_search_param.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/eigen.h"
#include "cupoch/utility/eigenvalue.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace registration {

namespace {

/// Obtain the Rotation matrix that transform the basis vector e1 onto the
/// input vector x.
inline __device__ Eigen::Matrix3f GetRotationFromE1ToX(const Eigen::Vector3f &x) {
    const Eigen::Vector3f e1{1, 0, 0};
    const Eigen::Vector3f v = e1.cross(x);
    const float c = e1.dot(x);
    if (c < -0.99) {
        // Then means that x and e1 are in the same direction
        return Eigen::Matrix3f::Identity();
    }

    const Eigen::Matrix3f sv = utility::SkewMatrix(v);
    const float factor = 1 / (1 + c);
    return Eigen::Matrix3f::Identity() + sv + (sv * sv) * factor;
}

/// Compute the covariance matrix according to the original paper. If the input
/// has already pre-computed covariances returns immediately. If the input has
/// pre-computed normals but no covariances, compute the covariances from those
/// normals. If there is no covariances nor normals, compute each covariance
/// matrix following the original implementation of GICP using 20 NN.
std::shared_ptr<geometry::PointCloud> InitializePointCloudForGeneralizedICP(
        const geometry::PointCloud &pcd, float epsilon) {
    auto output = std::make_shared<geometry::PointCloud>(pcd);
    if (output->HasCovariances()) {
        utility::LogDebug("GeneralizedICP: Using pre-computed covariances.");
        return output;
    }
    if (output->HasNormals()) {
        utility::LogDebug("GeneralizedICP: Computing covariances from normals");
    } else {
        // Compute covariances the same way is done in the original GICP paper.
        utility::LogDebug("GeneralizedICP: Computing covariances from points.");
        output->EstimateNormals(cupoch::knn::KDTreeSearchParamKNN(20));
    }

    output->covariances_.resize(output->points_.size());
    const Eigen::Matrix3f C = Eigen::Vector3f(epsilon, 1, 1).asDiagonal();
    thrust::transform(output->normals_.begin(), output->normals_.end(),
                      output->covariances_.begin(),
                      [C] __device__ (const Eigen::Vector3f &n) {
                          const auto Rx = GetRotationFromE1ToX(n);
                          return Rx * C * Rx.transpose();
                      });
    return output;
}

struct compute_jacobian_and_residual_functor
    : public utility::multiple_jacobians_residuals_functor<Eigen::Vector6f, 3> {
    compute_jacobian_and_residual_functor(
            const Eigen::Vector3f *source_points,
            const Eigen::Matrix3f *source_covariances,
            const Eigen::Vector3f *target_points,
            const Eigen::Matrix3f *target_covariances,
            const Eigen::Vector2i *corres)
        : source_points_(source_points),
          source_covariances_(source_covariances),
          target_points_(target_points),
          target_covariances_(target_covariances),
          corres_(corres) {};
    const Eigen::Vector3f *source_points_;
    const Eigen::Matrix3f *source_covariances_;
    const Eigen::Vector3f *target_points_;
    const Eigen::Matrix3f *target_covariances_;
    const Eigen::Vector2i *corres_;
    __device__ void operator()(int i,
                               Eigen::Vector6f J_r[3],
                               float r[3]) const {
        size_t cs = corres_[i][0];
        size_t ct = corres_[i][1];
        const Eigen::Vector3f &vs = source_points_[cs];
        const Eigen::Matrix3f &Cs = source_covariances_[cs];
        const Eigen::Vector3f &vt = target_points_[ct];
        const Eigen::Matrix3f &Ct = target_covariances_[ct];
        const Eigen::Vector3f d = vs - vt;
        Eigen::Matrix3f M_inv = (Ct + Cs).inverse();
        const Eigen::Matrix3f W = utility::SqrtMatrix3x3(M_inv);

        Eigen::Matrix<float, 3, 6> J;
        J.block<3, 3>(0, 0) = -utility::SkewMatrix(vs);
        J.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity();
        J = W * J;

        constexpr int n_rows = 3;
        for (size_t i = 0; i < n_rows; ++i) {
            r[i] = W.row(i).dot(d);
            J_r[i] = J.row(i);
        }
    }
};

struct mahalanobis_distance_functor {
    mahalanobis_distance_functor(
            const Eigen::Vector3f *source_points,
            const Eigen::Matrix3f *source_covariances,
            const Eigen::Vector3f *target_points,
            const Eigen::Matrix3f *target_covariances)
        : source_points_(source_points),
          source_covariances_(source_covariances),
          target_points_(target_points),
          target_covariances_(target_covariances) {};
    const Eigen::Vector3f *source_points_;
    const Eigen::Matrix3f *source_covariances_;
    const Eigen::Vector3f *target_points_;
    const Eigen::Matrix3f *target_covariances_;
    __device__ float operator()(const Eigen::Vector2i& cor) const {
        const Eigen::Vector3f &vs = source_points_[cor[0]];
        const Eigen::Matrix3f &Cs = source_covariances_[cor[0]];
        const Eigen::Vector3f &vt = target_points_[cor[1]];
        const Eigen::Matrix3f &Ct = target_covariances_[cor[1]];
        const Eigen::Vector3f d = vs - vt;
        Eigen::Matrix3f M_inv = (Ct + Cs).inverse();
        const Eigen::Matrix3f W = utility::SqrtMatrix3x3(M_inv);
        return d.transpose() * W * d;
    }
};

}  // namespace

float TransformationEstimationForGeneralizedICP::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty()) {
        return 0.0;
    }
    mahalanobis_distance_functor func(
        thrust::raw_pointer_cast(source.points_.data()),
        thrust::raw_pointer_cast(source.covariances_.data()),
        thrust::raw_pointer_cast(target.points_.data()),
        thrust::raw_pointer_cast(target.covariances_.data()));
    float err = thrust::transform_reduce(
            corres.begin(), corres.end(), func, 0.0f, thrust::plus<float>());
    return std::sqrt(err / (float)corres.size());
}

Eigen::Matrix4f
TransformationEstimationForGeneralizedICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || !target.HasCovariances() ||
        !source.HasCovariances()) {
        return Eigen::Matrix4f::Identity();
    }

    compute_jacobian_and_residual_functor func(
        thrust::raw_pointer_cast(source.points_.data()),
        thrust::raw_pointer_cast(source.covariances_.data()),
        thrust::raw_pointer_cast(target.points_.data()),
        thrust::raw_pointer_cast(target.covariances_.data()),
        thrust::raw_pointer_cast(corres.data()));

    Eigen::Matrix6f JTJ;
    Eigen::Vector6f JTr;
    float r2 = -1.0;
    thrust::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<
            Eigen::Matrix6f, Eigen::Vector6f, 3, compute_jacobian_and_residual_functor>(
                    func, (int)corres.size());

    bool is_success = false;
    Eigen::Matrix4f extrinsic;
    thrust::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Eigen::Matrix4f::Identity();
}

RegistrationResult RegistrationGeneralizedICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        float max_correspondence_distance,
        const Eigen::Matrix4f &init /* = Eigen::Matrix4f::Identity()*/,
        const TransformationEstimationForGeneralizedICP
                &estimation /* = TransformationEstimationForGeneralizedICP()*/,
        const ICPConvergenceCriteria
                &criteria /* = ICPConvergenceCriteria()*/) {
    return RegistrationICP(
            *InitializePointCloudForGeneralizedICP(source, estimation.epsilon_),
            *InitializePointCloudForGeneralizedICP(target, estimation.epsilon_),
            max_correspondence_distance, init, estimation, criteria);
}

}  // namespace registration
}  // namespace cupoch