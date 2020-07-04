#include "cupoch/geometry/pointcloud.h"
#include "cupoch/registration/filterreg.h"
#include "cupoch/registration/kabsch.h"
#include "cupoch/registration/permutohedral.h"

namespace cupoch {
namespace registration {

namespace {

struct weighted_residual_functor {
    __device__ float operator()(
            const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, float> &x) {
        const Eigen::Vector3f r =
                thrust::get<2>(x) * (thrust::get<0>(x) - thrust::get<1>(x));
        return r.squaredNorm();
    }
};

FilterRegResult GetRegistrationResult(
        const utility::device_vector<Eigen::Vector3f> &model,
        const utility::device_vector<Eigen::Vector3f> &target,
        const utility::device_vector<float> &weights,
        const Eigen::Matrix4f &transformation) {
    FilterRegResult result(transformation);
    result.likelihood_ = thrust::transform_reduce(
            make_tuple_begin(model, target, weights),
            make_tuple_end(model, target, weights), weighted_residual_functor(),
            0.0f, thrust::plus<float>());
    return result;
}

}  // namespace

FilterRegResult RegistrationFilterReg(const geometry::PointCloud &source,
                                      const geometry::PointCloud &target,
                                      const Eigen::Matrix4f &init,
                                      const FilterRegOption &option) {
    Eigen::Matrix4f transform = init;
    geometry::PointCloud model = source;
    if (init.isIdentity() == false) {
        model.Transform(init);
    }
    FilterRegResult result(init);
    utility::device_vector<Eigen::Vector3f> target_pt(source.points_.size(),
                                                      Eigen::Vector3f::Zero());
    utility::device_vector<float> weight(source.points_.size(), 0.0f);
    utility::device_vector<float> m2(source.points_.size(), 0.0f);
    registration::Permutohedral<3> pmh(option.sigma_initial_);
    pmh.BuildLatticeIndexNoBlur(target.points_, target.points_);
    for (int i = 0; i < option.max_iteration_; ++i) {
        // Compute target
        pmh.ComputeTarget(model.points_, target_pt, weight, m2);
        Eigen::Matrix4f update =
                registration::KabschWeighted(model.points_, target_pt, weight);
        transform = update * transform;
        model.Transform(update);
        const auto sigma =
                pmh.ComputeSigma(model.points_, target_pt, weight, m2);
        if (!std::isnan(sigma) && sigma > option.sigma_min_) {
            pmh.sigma_ = Eigen::Vector3f::Constant(sigma);
            pmh.lattice_map_.clear();
            pmh.BuildLatticeIndexNoBlur(target.points_, target.points_);
        }
        FilterRegResult backup = result;
        result = GetRegistrationResult(model.points_, target_pt, weight,
                                       transform);
        if (std::abs(backup.likelihood_ - result.likelihood_) <
            option.relative_likelihood_) {
            break;
        }
    }
    return result;
}

}  // namespace registration
}  // namespace cupoch