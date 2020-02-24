#include <Eigen/Geometry>
#include <iostream>

#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/geometry/kdtree_search_param.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/registration/colored_icp.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/eigen.h"

using namespace cupoch;
using namespace cupoch::registration;

namespace {

class PointCloudForColoredICP : public geometry::PointCloud {
public:
    utility::device_vector<Eigen::Vector3f> color_gradient_;
};

class TransformationEstimationForColoredICP : public TransformationEstimation {
public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    TransformationEstimationForColoredICP(float lambda_geometric = 0.968)
        : lambda_geometric_(lambda_geometric) {
        if (lambda_geometric_ < 0 || lambda_geometric_ > 1.0)
            lambda_geometric_ = 0.968;
    }
    ~TransformationEstimationForColoredICP() override {}

public:
    float ComputeRMSE(const geometry::PointCloud &source,
                      const geometry::PointCloud &target,
                      const CorrespondenceSet &corres) const override;
    Eigen::Matrix4f ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

public:
    float lambda_geometric_;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::ColoredICP;
};

struct compute_color_gradient_functor {
    compute_color_gradient_functor(const Eigen::Vector3f *points,
                                   const Eigen::Vector3f *normals,
                                   const Eigen::Vector3f *colors,
                                   const int *indices,
                                   const float *distances2,
                                   int knn)
        : points_(points),
          normals_(normals),
          colors_(colors),
          indices_(indices),
          distances2_(distances2),
          knn_(knn){};
    const Eigen::Vector3f *points_;
    const Eigen::Vector3f *normals_;
    const Eigen::Vector3f *colors_;
    const int *indices_;
    const float *distances2_;
    const int knn_;
    __device__ Eigen::Vector3f operator()(size_t idx) const {
        const Eigen::Vector3f &vt = points_[idx];
        const Eigen::Vector3f &nt = normals_[idx];
        float it = (colors_[idx](0) + colors_[idx](1) + colors_[idx](2)) / 3.0;
        Eigen::Matrix<float, geometry::NUM_MAX_NN, 3> A;
        Eigen::Matrix<float, geometry::NUM_MAX_NN, 1> b;
        A.setZero();
        b.setZero();
        int nn = 0;
        for (size_t i = 1; i < knn_; ++i) {
            if (indices_[idx * knn_ + i] < 0) continue;
            int P_adj_idx = indices_[idx * knn_ + i];
            Eigen::Vector3f vt_adj = points_[P_adj_idx];
            Eigen::Vector3f vt_proj = vt_adj - (vt_adj - vt).dot(nt) * nt;
            float it_adj = (colors_[P_adj_idx](0) + colors_[P_adj_idx](1) +
                            colors_[P_adj_idx](2)) /
                           3.0;
            A(nn - 1, 0) = (vt_proj(0) - vt(0));
            A(nn - 1, 1) = (vt_proj(1) - vt(1));
            A(nn - 1, 2) = (vt_proj(2) - vt(2));
            b(nn - 1, 0) = (it_adj - it);
            ++nn;
        }
        if (nn < 4) return Eigen::Vector3f::Zero();
        // adds orthogonal constraint
        A(nn - 1, 0) = (nn - 1) * nt(0);
        A(nn - 1, 1) = (nn - 1) * nt(1);
        A(nn - 1, 2) = (nn - 1) * nt(2);
        b(nn - 1, 0) = 0;
        // solving linear equation
        Eigen::Matrix3f AtA = A.transpose() * A;
        Eigen::Vector3f Atb = A.transpose() * b;
        const Eigen::Vector3f x = AtA.inverse() * Atb;
        return x;
    }
};

std::shared_ptr<PointCloudForColoredICP> InitializePointCloudForColoredICP(
        const geometry::PointCloud &target,
        const geometry::KDTreeSearchParamHybrid &search_param) {
    utility::LogDebug("InitializePointCloudForColoredICP");

    geometry::KDTreeFlann tree;
    tree.SetGeometry(target);

    auto output = std::make_shared<PointCloudForColoredICP>();
    output->colors_ = target.colors_;
    output->normals_ = target.normals_;
    output->points_ = target.points_;

    size_t n_points = output->points_.size();
    output->color_gradient_.resize(n_points, Eigen::Vector3f::Zero());
    utility::device_vector<int> point_idx;
    utility::device_vector<float> point_squared_distance;
    tree.SearchHybrid(output->points_, search_param.radius_,
                      search_param.max_nn_, point_idx, point_squared_distance);
    compute_color_gradient_functor func(
            thrust::raw_pointer_cast(output->points_.data()),
            thrust::raw_pointer_cast(output->normals_.data()),
            thrust::raw_pointer_cast(output->colors_.data()),
            thrust::raw_pointer_cast(point_idx.data()),
            thrust::raw_pointer_cast(point_squared_distance.data()),
            search_param.max_nn_);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_points),
                      output->color_gradient_.begin(), func);
    return output;
}

struct compute_jacobian_and_residual_functor
    : public utility::multiple_jacobians_residuals_functor<Eigen::Vector6f, 2> {
    compute_jacobian_and_residual_functor(
            const Eigen::Vector3f *source_points,
            const Eigen::Vector3f *source_colors,
            const Eigen::Vector3f *target_points,
            const Eigen::Vector3f *target_normals,
            const Eigen::Vector3f *target_colors,
            const Eigen::Vector3f *target_color_gradient,
            const Eigen::Vector2i *corres,
            float sqrt_lambda_geometric,
            float sqrt_lambda_photometric)
        : source_points_(source_points),
          source_colors_(source_colors),
          target_points_(target_points),
          target_normals_(target_normals),
          target_colors_(target_colors),
          target_color_gradient_(target_color_gradient),
          corres_(corres),
          sqrt_lambda_geometric_(sqrt_lambda_geometric),
          sqrt_lambda_photometric_(sqrt_lambda_photometric){};
    const Eigen::Vector3f *source_points_;
    const Eigen::Vector3f *source_colors_;
    const Eigen::Vector3f *target_points_;
    const Eigen::Vector3f *target_normals_;
    const Eigen::Vector3f *target_colors_;
    const Eigen::Vector3f *target_color_gradient_;
    const Eigen::Vector2i *corres_;
    const float sqrt_lambda_geometric_;
    const float sqrt_lambda_photometric_;
    __device__ void operator()(int i,
                               Eigen::Vector6f J_r[2],
                               float r[2]) const {
        size_t cs = corres_[i][0];
        size_t ct = corres_[i][1];
        const Eigen::Vector3f &vs = source_points_[cs];
        const Eigen::Vector3f &vt = target_points_[ct];
        const Eigen::Vector3f &nt = target_normals_[ct];

        J_r[0].block<3, 1>(0, 0) = sqrt_lambda_geometric_ * vs.cross(nt);
        J_r[0].block<3, 1>(3, 0) = sqrt_lambda_geometric_ * nt;
        r[0] = sqrt_lambda_geometric_ * (vs - vt).dot(nt);

        // project vs into vt's tangential plane
        Eigen::Vector3f vs_proj = vs - (vs - vt).dot(nt) * nt;
        float is = (source_colors_[cs](0) + source_colors_[cs](1) +
                    source_colors_[cs](2)) /
                   3.0;
        float it = (target_colors_[ct](0) + target_colors_[ct](1) +
                    target_colors_[ct](2)) /
                   3.0;
        const Eigen::Vector3f &dit = target_color_gradient_[ct];
        float is0_proj = (dit.dot(vs_proj - vt)) + it;

        const Eigen::Matrix3f M =
                (Eigen::Matrix3f() << 1.0 - nt(0) * nt(0), -nt(0) * nt(1),
                 -nt(0) * nt(2), -nt(0) * nt(1), 1.0 - nt(1) * nt(1),
                 -nt(1) * nt(2), -nt(0) * nt(2), -nt(1) * nt(2),
                 1.0 - nt(2) * nt(2))
                        .finished();

        const Eigen::Vector3f &ditM = -dit.transpose() * M;
        J_r[1].block<3, 1>(0, 0) = sqrt_lambda_photometric_ * vs.cross(ditM);
        J_r[1].block<3, 1>(3, 0) = sqrt_lambda_photometric_ * ditM;
        r[1] = sqrt_lambda_photometric_ * (is - is0_proj);
    }
};

Eigen::Matrix4f TransformationEstimationForColoredICP::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || target.HasNormals() == false ||
        target.HasColors() == false || source.HasColors() == false)
        return Eigen::Matrix4f::Identity();

    float sqrt_lambda_geometric = sqrt(lambda_geometric_);
    float lambda_photometric = 1.0 - lambda_geometric_;
    float sqrt_lambda_photometric = sqrt(lambda_photometric);

    const auto &target_c = (const PointCloudForColoredICP &)target;

    compute_jacobian_and_residual_functor func(
            thrust::raw_pointer_cast(source.points_.data()),
            thrust::raw_pointer_cast(source.colors_.data()),
            thrust::raw_pointer_cast(target.points_.data()),
            thrust::raw_pointer_cast(target.normals_.data()),
            thrust::raw_pointer_cast(target.colors_.data()),
            thrust::raw_pointer_cast(target_c.color_gradient_.data()),
            thrust::raw_pointer_cast(corres.data()), sqrt_lambda_geometric,
            sqrt_lambda_photometric);
    Eigen::Matrix6f JTJ;
    Eigen::Vector6f JTr;
    float r2;
    thrust::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix6f, Eigen::Vector6f, 2,
                                      compute_jacobian_and_residual_functor>(
                    func, (int)corres.size());

    bool is_success;
    Eigen::Matrix4f extrinsic;
    thrust::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Eigen::Matrix4f::Identity();
}

struct diff_square_colored_functor {
    diff_square_colored_functor(const Eigen::Vector3f *source_points,
                                const Eigen::Vector3f *source_colors,
                                const Eigen::Vector3f *target_points,
                                const Eigen::Vector3f *target_normals,
                                const Eigen::Vector3f *target_colors,
                                const Eigen::Vector3f *target_color_gradient,
                                const Eigen::Vector2i *corres,
                                float sqrt_lambda_geometric,
                                float sqrt_lambda_photometric)
        : source_points_(source_points),
          source_colors_(source_colors),
          target_points_(target_points),
          target_normals_(target_normals),
          target_colors_(target_colors),
          target_color_gradient_(target_color_gradient),
          corres_(corres),
          sqrt_lambda_geometric_(sqrt_lambda_geometric),
          sqrt_lambda_photometric_(sqrt_lambda_photometric){};
    const Eigen::Vector3f *source_points_;
    const Eigen::Vector3f *source_colors_;
    const Eigen::Vector3f *target_points_;
    const Eigen::Vector3f *target_normals_;
    const Eigen::Vector3f *target_colors_;
    const Eigen::Vector3f *target_color_gradient_;
    const Eigen::Vector2i *corres_;
    const float sqrt_lambda_geometric_;
    const float sqrt_lambda_photometric_;
    __device__ float operator()(size_t idx) const {
        size_t cs = corres_[idx][0];
        size_t ct = corres_[idx][1];
        const Eigen::Vector3f &vs = source_points_[cs];
        const Eigen::Vector3f &vt = target_points_[ct];
        const Eigen::Vector3f &nt = target_normals_[ct];
        Eigen::Vector3f vs_proj = vs - (vs - vt).dot(nt) * nt;
        float is = (source_colors_[cs](0) + source_colors_[cs](1) +
                    source_colors_[cs](2)) /
                   3.0;
        float it = (target_colors_[ct](0) + target_colors_[ct](1) +
                    target_colors_[ct](2)) /
                   3.0;
        const Eigen::Vector3f &dit = target_color_gradient_[ct];
        float is0_proj = (dit.dot(vs_proj - vt)) + it;
        float residual_geometric = sqrt_lambda_geometric_ * (vs - vt).dot(nt);
        float residual_photometric = sqrt_lambda_photometric_ * (is - is0_proj);
        return residual_geometric * residual_geometric +
               residual_photometric * residual_photometric;
    }
};

float TransformationEstimationForColoredICP::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    float sqrt_lambda_geometric = sqrt(lambda_geometric_);
    float lambda_photometric = 1.0 - lambda_geometric_;
    float sqrt_lambda_photometric = sqrt(lambda_photometric);
    const auto &target_c = (const PointCloudForColoredICP &)target;

    diff_square_colored_functor func(
            thrust::raw_pointer_cast(source.points_.data()),
            thrust::raw_pointer_cast(source.colors_.data()),
            thrust::raw_pointer_cast(target.points_.data()),
            thrust::raw_pointer_cast(target.normals_.data()),
            thrust::raw_pointer_cast(target.colors_.data()),
            thrust::raw_pointer_cast(target_c.color_gradient_.data()),
            thrust::raw_pointer_cast(corres.data()), sqrt_lambda_geometric,
            sqrt_lambda_photometric);
    const auto err = thrust::transform_reduce(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator(corres.size()), func, 0.0f,
            thrust::plus<float>());
    return err;
};

}  // namespace

RegistrationResult cupoch::registration::RegistrationColoredICP(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        float max_distance,
        const Eigen::Matrix4f &init /* = Eigen::Matrix4f::Identity()*/,
        const ICPConvergenceCriteria &criteria /* = ICPConvergenceCriteria()*/,
        float lambda_geometric /* = 0.968*/) {
    auto target_c = InitializePointCloudForColoredICP(
            target, geometry::KDTreeSearchParamHybrid(max_distance * 2.0, 30));
    return RegistrationICP(
            source, *target_c, max_distance, init,
            TransformationEstimationForColoredICP(lambda_geometric), criteria);
}