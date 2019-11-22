#include "cupoc/registration/registration.h"
#include "cupoc/geometry/kdtree_flann.h"
#include "cupoc/utility/helper.h"
#include "cupoc/utility/console.h"

using namespace cupoc;
using namespace cupoc::registration;

namespace {

struct extact_knn_distance_functor {
    __device__
    float operator() (const geometry::KNNDistances& x) const {
        return max(x[0], 0.0);
    }
};

struct make_correspondence_pair_functor {
   __device__
   thrust::tuple<int, int> operator() (int i, const geometry::KNNIndices& idxs) const {
        if (idxs[0] < 0) {
            return thrust::make_tuple(-1, -1);
        } else {
            return thrust::make_tuple(i, idxs[0]);
        }
   }
};

RegistrationResult GetRegistrationResultAndCorrespondences(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const geometry::KDTreeFlann &target_kdtree,
    float max_correspondence_distance,
    const Eigen::Matrix4f &transformation) {
    RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0) {
        return result;
    }

    const int n_pt = source.points_.size();
    thrust::device_vector<geometry::KNNIndices> indices(n_pt);
    thrust::device_vector<geometry::KNNDistances> dists(n_pt);
    target_kdtree.SearchHybrid(source.points_, max_correspondence_distance,
                               1, indices, dists);
    const float error2 = thrust::transform_reduce(dists.begin(), dists.end(),
                                                  extact_knn_distance_functor(),
                                                  0.0f, thrust::plus<float>());
    result.correspondence_set_.resize(n_pt);
    thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(n_pt),
                      indices.begin(), result.correspondence_set_.begin(),
                      make_correspondence_pair_functor());
    auto end = thrust::remove_if(result.correspondence_set_.begin(), result.correspondence_set_.end(),
                                 [] __device__ (const thrust::tuple<int, int>& x) -> bool {return (thrust::get<0>(x) < 0);});
    int n_out = static_cast<int>(end - result.correspondence_set_.begin());
    result.correspondence_set_.resize(n_out);

    if (result.correspondence_set_.empty()) {
        result.fitness_ = 0.0;
        result.inlier_rmse_ = 0.0;
    } else {
        size_t corres_number = result.correspondence_set_.size();
        result.fitness_ = (float)corres_number / (float)source.points_.size();
        result.inlier_rmse_ = std::sqrt(error2 / (float)corres_number);
    }
    return result;
}

}

RegistrationResult cupoc::registration::RegistrationICP(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    float max_correspondence_distance,
    const Eigen::Matrix4f &init /* = Eigen::Matrix4f::Identity()*/,
    const TransformationEstimation &estimation
    /* = TransformationEstimationPointToPoint(false)*/,
    const ICPConvergenceCriteria
            &criteria /* = ICPConvergenceCriteria()*/) {
    if (max_correspondence_distance <= 0.0) {
        utility::LogError("Invalid max_correspondence_distance.");
    }

    if ((estimation.GetTransformationEstimationType() ==
                TransformationEstimationType::PointToPlane ||
         estimation.GetTransformationEstimationType() ==
                TransformationEstimationType::ColoredICP) &&
        (!source.HasNormals() || !target.HasNormals())) {
        utility::LogError(
                "TransformationEstimationPointToPlane and "
                "TransformationEstimationColoredICP "
                "require pre-computed normal vectors.");
    }

    Eigen::Matrix4f transformation = init;
    geometry::KDTreeFlann kdtree(target);
    geometry::PointCloud pcd = source;
    if (init.isIdentity() == false) {
        pcd.Transform(init);
    }
    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);
    for (int i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}", i,
                          result.fitness_, result.inlier_rmse_);
        Eigen::Matrix4f update = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_);
        transformation = update * transformation;
        pcd.Transform(update);
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                pcd, target, kdtree, max_correspondence_distance,
                transformation);
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            break;
        }
    }
    return result;
}