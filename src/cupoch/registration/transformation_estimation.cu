#include "cupoch/registration/transformation_estimation.h"
#include "cupoch/registration/kabsch.h"
#include "cupoch/geometry/pointcloud.h"
#include <Eigen/Geometry>
#include <thrust/transform_reduce.h>

using namespace cupoch;
using namespace cupoch::registration;

namespace{

struct diff_square_pt2pt_functor {
    diff_square_pt2pt_functor(const Eigen::Vector3f* source,
                              const Eigen::Vector3f* target,
                              const Eigen::Vector2i* corres)
        : source_(source), target_(target), corres_(corres) {};
    const Eigen::Vector3f* source_;
    const Eigen::Vector3f* target_;
    const Eigen::Vector2i* corres_;
    __device__
    float operator()(size_t idx) const {
        return (source_[corres_[idx][0]] - target_[corres_[idx][1]]).squaredNorm();
    }
};

struct diff_square_pt2pl_functor {
    diff_square_pt2pl_functor(const Eigen::Vector3f* source,
                              const Eigen::Vector3f* target_points,
                              const Eigen::Vector3f* target_normals,
                              const Eigen::Vector2i* corres)
        : source_(source), target_points_(target_points), target_normals_(target_normals), corres_(corres) {};
    const Eigen::Vector3f* source_;
    const Eigen::Vector3f* target_points_;
    const Eigen::Vector3f* target_normals_;
    const Eigen::Vector2i* corres_;
    __device__
    float operator()(size_t idx) const {
        float r = (source_[corres_[idx][0]] - target_points_[corres_[idx][1]]).dot(target_normals_[corres_[idx][1]]);
        return r * r;
    }
};

struct pt2pl_jacobian_residual_functor : public utility::jacobian_residual_functor<Eigen::Vector6f> {
    pt2pl_jacobian_residual_functor(const Eigen::Vector3f* source,
                                    const Eigen::Vector3f* target_points,
                                    const Eigen::Vector3f* target_normals,
                                    const Eigen::Vector2i* corres)
        : source_(source), target_points_(target_points), target_normals_(target_normals) {};
    const Eigen::Vector3f* source_;
    const Eigen::Vector3f* target_points_;
    const Eigen::Vector3f* target_normals_;
    const Eigen::Vector2i* corres_;
    __device__
    void operator() (int idx, Eigen::Vector6f& vec, float& r) const {
        const Eigen::Vector3f &vs = source_[corres_[idx][0]];
        const Eigen::Vector3f &vt = target_points_[corres_[idx][1]];
        const Eigen::Vector3f &nt = target_normals_[corres_[idx][1]];
        r = (vs - vt).dot(nt);
        vec.block<3, 1>(0, 0) = vs.cross(nt);
        vec.block<3, 1>(3, 0) = nt;
    }
};

}

float TransformationEstimationPointToPoint::ComputeRMSE(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const CorrespondenceSet &corres) const {
    diff_square_pt2pt_functor func(thrust::raw_pointer_cast(source.points_.data()),
                                   thrust::raw_pointer_cast(target.points_.data()),
                                   thrust::raw_pointer_cast(corres.data()));
    const float err = thrust::transform_reduce(thrust::make_counting_iterator<size_t>(0),
                                               thrust::make_counting_iterator(corres.size()),
                                               func, 0.0f, thrust::plus<float>());
    return std::sqrt(err / (float)corres.size());
}

Eigen::Matrix4f TransformationEstimationPointToPoint::ComputeTransformation(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const CorrespondenceSet &corres) const {
    return Kabsch(source.points_, target.points_, corres);
}

float TransformationEstimationPointToPlane::ComputeRMSE(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const CorrespondenceSet &corres) const {
    if (corres.empty() || !target.HasNormals()) return 0.0;
    diff_square_pt2pl_functor func(thrust::raw_pointer_cast(source.points_.data()),
                                   thrust::raw_pointer_cast(target.points_.data()),
                                   thrust::raw_pointer_cast(target.normals_.data()),
                                   thrust::raw_pointer_cast(corres.data()));
    const float err = thrust::transform_reduce(thrust::make_counting_iterator<size_t>(0),
                                               thrust::make_counting_iterator(corres.size()),
                                               func, 0.0f, thrust::plus<float>());
    return std::sqrt(err / (float)corres.size());
}

Eigen::Matrix4f TransformationEstimationPointToPlane::ComputeTransformation(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const CorrespondenceSet &corres) const {
    if (corres.empty() || !target.HasNormals()) return Eigen::Matrix4f::Identity();

    Eigen::Matrix6f JTJ;
    Eigen::Vector6f JTr;
    float r2;
    pt2pl_jacobian_residual_functor func(thrust::raw_pointer_cast(source.points_.data()),
                                         thrust::raw_pointer_cast(target.points_.data()),
                                         thrust::raw_pointer_cast(target.normals_.data()),
                                         thrust::raw_pointer_cast(corres.data()));
    thrust::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix6f, Eigen::Vector6f>(
                func, (int)corres.size());

    bool is_success;
    Eigen::Matrix4f extrinsic;
    thrust::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);

    return is_success ? extrinsic : Eigen::Matrix4f::Identity();
}