#include "cupoch/registration/registration.h"
#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/utility/helper.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/platform.h"

using namespace cupoch;
using namespace cupoch::registration;

namespace {

struct extact_knn_distance_functor {
    extact_knn_distance_functor(const float* distances) : distances_(distances) {};
    const float* distances_;
    __device__
    float operator() (int idx) const {
        return (std::isinf(distances_[idx])) ? 0.0 : distances_[idx];
    }
};

struct make_correspondence_pair_functor {
    make_correspondence_pair_functor(const int* indices) : indices_(indices) {};
    const int* indices_;
   __device__
   Eigen::Vector2i operator() (int i) const {
        return (indices_[i] < 0) ? Eigen::Vector2i(-1, -1) : Eigen::Vector2i(i, indices_[i]);
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
    thrust::device_vector<int> indices(n_pt);
    thrust::device_vector<float> dists(n_pt);
    target_kdtree.SearchHybrid(source.points_, max_correspondence_distance,
                               1, indices, dists);
    extact_knn_distance_functor func(thrust::raw_pointer_cast(dists.data()));
    result.correspondence_set_.resize(n_pt);
    const float error2 = thrust::transform_reduce(thrust::cuda::par.on(utility::GetStream(0)),
                                                  thrust::make_counting_iterator(0),
                                                  thrust::make_counting_iterator(n_pt),
                                                  func, 0.0f, thrust::plus<float>());
    thrust::transform(thrust::cuda::par.on(utility::GetStream(1)),
                      thrust::make_counting_iterator(0), thrust::make_counting_iterator(n_pt),
                      result.correspondence_set_.begin(),
                      make_correspondence_pair_functor(thrust::raw_pointer_cast(indices.data())));
    auto end = thrust::remove_if(thrust::cuda::par.on(utility::GetStream(1)),
                                 result.correspondence_set_.begin(), result.correspondence_set_.end(),
                                 [] __device__ (const Eigen::Vector2i& x) -> bool {return (x[0] < 0);});
    cudaStreamSynchronize(utility::GetStream(1));
    int n_out = thrust::distance(result.correspondence_set_.begin(), end);
    result.correspondence_set_.resize(n_out);
    cudaDeviceSynchronize();

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

RegistrationResult::RegistrationResult(const Eigen::Matrix4f &transformation)
    : transformation_(transformation), inlier_rmse_(0.0), fitness_(0.0) {}

RegistrationResult::RegistrationResult(const RegistrationResult& other)
    : transformation_(other.transformation_), correspondence_set_(other.correspondence_set_), inlier_rmse_(other.inlier_rmse_), fitness_(other.fitness_)
{}

RegistrationResult::~RegistrationResult() {}

void RegistrationResult::SetCorrespondenceSet(const thrust::host_vector<Eigen::Vector2i>& corres) {
    correspondence_set_ = corres;
}

thrust::host_vector<Eigen::Vector2i> RegistrationResult::GetCorrespondenceSet() const {
    thrust::host_vector<Eigen::Vector2i> corres = correspondence_set_;
    return corres;
}


RegistrationResult cupoch::registration::RegistrationICP(
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