/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/
#include "cupoch/knn/kdtree_flann.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/geometry_utils.h"
#include "cupoch/registration/registration.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/helper.h"
#include "cupoch/utility/random.h"

#include <omp.h>

using namespace cupoch;
using namespace cupoch::registration;

namespace {

RegistrationResult GetRegistrationResultAndCorrespondences(
        cudaStream_t stream,
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const knn::KDTreeFlann &target_kdtree,
        float max_correspondence_distance,
        const Eigen::Matrix4f &transformation) {
    RegistrationResult result(transformation);
    if (max_correspondence_distance <= 0.0) {
        return result;
    }

    const int n_pt = source.points_.size();
    utility::device_vector<int> indices(n_pt);
    utility::device_vector<float> dists(n_pt);
    target_kdtree.SearchRadius(source.points_, max_correspondence_distance, 1,
                               indices, dists);
    result.correspondence_set_.resize(n_pt);
    const float error2 = thrust::transform_reduce(
            utility::exec_policy(stream), dists.begin(), dists.end(),
            [] __device__(float d) { return (isinf(d)) ? 0.0 : d; }, 0.0f,
            thrust::plus<float>());
    thrust::transform(utility::exec_policy(stream),
                      enumerate_begin(indices), enumerate_end(indices),
                      result.correspondence_set_.begin(),
                      [] __device__(const thrust::tuple<int, int> &idxs) {
                          int j = thrust::get<1>(idxs);
                          return (j < 0) ? Eigen::Vector2i(-1, -1)
                                         : Eigen::Vector2i(thrust::get<0>(idxs),
                                                           j);
                      });
    auto end =
            thrust::remove_if(utility::exec_policy(stream),
                              result.correspondence_set_.begin(),
                              result.correspondence_set_.end(),
                              [] __device__(const Eigen::Vector2i &x) -> bool {
                                  return (x[0] < 0);
                              });
    int n_out = thrust::distance(result.correspondence_set_.begin(), end);
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

struct correspondence_count_functor {
    const Eigen::Vector3f *source_points_;
    float max_dis2_;
    correspondence_count_functor(float max_correspondence_distance)
        : max_dis2_(max_correspondence_distance * max_correspondence_distance) {}
    __device__ int operator()(const Eigen::Vector2i &c) const {
        float dis2 = (source_points_[c[0]] - source_points_[c[1]]).squaredNorm();
        return (dis2 < max_dis2_) ? 1 : 0;
    }
};

float EvaluateInlierCorrespondenceRatio(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        float max_correspondence_distance,
        const Eigen::Matrix4f &transformation) {
    int inlier_corres = thrust::transform_reduce(
            utility::exec_policy(0), corres.begin(), corres.end(),
            correspondence_count_functor(max_correspondence_distance), 0, thrust::plus<int>());
    return static_cast<float>(inlier_corres / corres.size());
}

}  // namespace

RegistrationResult::RegistrationResult(const Eigen::Matrix4f &transformation)
    : transformation_(transformation), inlier_rmse_(0.0), fitness_(0.0) {}

RegistrationResult::RegistrationResult(const RegistrationResult &other)
    : transformation_(other.transformation_),
      correspondence_set_(other.correspondence_set_),
      inlier_rmse_(other.inlier_rmse_),
      fitness_(other.fitness_) {}

RegistrationResult::~RegistrationResult() {}

void RegistrationResult::SetCorrespondenceSet(
        const thrust::host_vector<Eigen::Vector2i> &corres) {
    correspondence_set_ = corres;
}

thrust::host_vector<Eigen::Vector2i> RegistrationResult::GetCorrespondenceSet()
        const {
    thrust::host_vector<Eigen::Vector2i> corres = correspondence_set_;
    return corres;
}

RegistrationResult cupoch::registration::EvaluateRegistration(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        float max_correspondence_distance,
        const Eigen::Matrix4f
                &transformation /* = Eigen::Matrix4d::Identity()*/) {
    knn::KDTreeFlann kdtree(geometry::ConvertVector3fVectorRef(target));
    geometry::PointCloud pcd = source;
    if (!transformation.isIdentity()) {
        pcd.Transform(transformation);
    }
    return GetRegistrationResultAndCorrespondences(
            0, pcd, target, kdtree, max_correspondence_distance, transformation);
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
        !target.HasNormals()) {
        utility::LogError(
                "TransformationEstimationPointToPlane and "
                "TransformationEstimationColoredICP "
                "require pre-computed target normal vectors.");
    }

    Eigen::Matrix4f transformation = init;
    knn::KDTreeFlann kdtree(geometry::ConvertVector3fVectorRef(target));
    geometry::PointCloud pcd = source;
    if (init.isIdentity() == false) {
        pcd.Transform(init);
    }
    RegistrationResult result;
    result = GetRegistrationResultAndCorrespondences(
            0, pcd, target, kdtree, max_correspondence_distance, transformation);
    for (int i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}", i,
                          result.fitness_, result.inlier_rmse_);
        Eigen::Matrix4f update = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_);
        transformation = update * transformation;
        pcd.Transform(update);
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondences(
                0, pcd, target, kdtree, max_correspondence_distance,
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

RegistrationResult RegistrationRANSACBasedOnCorrespondence(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        float max_correspondence_distance,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        int ransac_n /* = 3*/,
        const std::vector<std::reference_wrapper<const CorrespondenceChecker>>
                &checkers /* = {}*/,
        const RANSACConvergenceCriteria &criteria
        /* = RANSACConvergenceCriteria()*/) {
    if (ransac_n < 3 || (int)corres.size() < ransac_n ||
        max_correspondence_distance <= 0.0) {
        return RegistrationResult();
    }
    RegistrationResult best_result;
    knn::KDTreeFlann kdtree(geometry::ConvertVector3fVectorRef(target));
    int est_k_global = criteria.max_iteration_;
    int total_validation = 0;

#pragma omp parallel
    {
        CorrespondenceSet ransac_corres(ransac_n);
        RegistrationResult best_result_local;
        int est_k_local = criteria.max_iteration_;
        utility::random::UniformIntGenerator<int> rand_gen(0,
                                                           corres.size() - 1);

#pragma omp for nowait
        for (int itr = 0; itr < criteria.max_iteration_; itr++) {
            if (itr >= est_k_global) {
                continue;
            }
            int thread_id = omp_get_thread_num();
            cudaStream_t stream1 = utility::GetStream((2 * thread_id) % utility::MAX_NUM_STREAMS);
            cudaStream_t stream2 = utility::GetStream((2 * thread_id + 1) % utility::MAX_NUM_STREAMS);
            // Randomly select ransac_n correspondences
            for (int i = 0; i < ransac_n; i++) {
                ransac_corres[i] = corres[rand_gen()];
            }
            // Estimate transformation
            Eigen::Matrix4f transformation = estimation.ComputeTransformation(
                    stream1, stream2,
                    source, target, ransac_corres);

            bool check = true;
            for (const auto &checker : checkers) {
                if (!checker.get().Check(source, target, ransac_corres,
                                         transformation)) {
                    check = false;
                    break;
                }
            }
            if (!check) continue;

            geometry::PointCloud pcd = source;
            pcd.Transform(stream1, stream2, transformation);
            RegistrationResult result = GetRegistrationResultAndCorrespondences(
                stream1, pcd, target, kdtree, max_correspondence_distance,
                transformation);
            if (result.IsBetterRANSACThan(best_result_local)) {
                best_result_local = result;

                float corres_inlier_ratio =
                        EvaluateInlierCorrespondenceRatio(
                                pcd, target, corres,
                                max_correspondence_distance,
                                transformation);

                // Update exit condition if necessary.
                // If confidence is 1.0, then it is safely inf, we always
                // consume all the iterations.
                float est_k_local_d =
                        std::log(1.0 - criteria.confidence_) /
                        std::log(1.0 -
                                 std::pow(corres_inlier_ratio, ransac_n));
                est_k_local =
                        est_k_local_d < est_k_global
                                ? static_cast<int>(std::ceil(est_k_local_d))
                                : est_k_local;
                utility::LogDebug(
                        "Thread {:06d}: registration fitness={:.3f}, "
                        "corres inlier ratio={:.3f}, "
                        "Est. max k = {}",
                        itr, result.fitness_, corres_inlier_ratio,
                        est_k_local_d);
                }
#pragma omp critical
            {
                total_validation += 1;
                if (est_k_local < est_k_global) {
                    est_k_global = est_k_local;
                }
            }
        }      // for loop
#pragma omp critical(RegistrationRANSACBasedOnCorrespondence)
        {
            if (best_result_local.IsBetterRANSACThan(best_result)) {
                best_result = best_result_local;
            }
        }
    }
    utility::LogDebug(
            "RANSAC exits after {:d} validations. Best inlier ratio {:e}, "
            "RMSE {:e}",
            total_validation, best_result.fitness_, best_result.inlier_rmse_);
    return best_result;
}