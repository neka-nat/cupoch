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
#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/registration/registration.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/helper.h"

using namespace cupoch;
using namespace cupoch::registration;

namespace {

    struct NormalRejector {
          NormalRejector(const Eigen::Vector3f *sourceNormal,
                         const Eigen::Vector3f *targetNormal,
                         float normal_angle_threshold_)
            : cos_normal_angle_threshold(cos(normal_angle_threshold_)),
              sourceNormal(sourceNormal),
              targetNormal(targetNormal){};
        
        __device__ bool operator()(const Eigen::Vector2i correspondence) const {
            
            Eigen::Vector3f trgN = targetNormal[correspondence[1]];
            Eigen::Vector3f srcN = sourceNormal[correspondence[0]];
            
            if (trgN.dot(srcN) > cos_normal_angle_threshold) {
                return false;
            }
            return true;
        }

        const Eigen::Vector3f *sourceNormal;
        const Eigen::Vector3f *targetNormal;
        const float cos_normal_angle_threshold;
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
    utility::device_vector<int> indices(n_pt);
    utility::device_vector<float> dists(n_pt);
    target_kdtree.SearchRadius(source.points_, max_correspondence_distance, 1,
                               indices, dists);
    result.correspondence_set_.resize(n_pt);
    const float error2 = thrust::transform_reduce(
            dists.begin(), dists.end(),
            [] __device__(float d) { return (isinf(d)) ? 0.0 : d; }, 0.0f,
            thrust::plus<float>());
    thrust::transform(enumerate_begin(indices), enumerate_end(indices),
                      result.correspondence_set_.begin(),
                      [] __device__(const thrust::tuple<int, int> &idxs) {
                          int j = thrust::get<1>(idxs);
                          return (j < 0) ? Eigen::Vector2i(-1, -1)
                                         : Eigen::Vector2i(thrust::get<0>(idxs),
                                                           j);
                      });
    auto end =
            thrust::remove_if(result.correspondence_set_.begin(),
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

RegistrationResult GetRegistrationResultAndCorrespondencesWithNormalRejection(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const geometry::KDTreeFlann &target_kdtree,
        float max_correspondence_distance,
        float max_normal_difference,
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
            dists.begin(), dists.end(),
            [] __device__(float d) { return (isinf(d)) ? 0.0 : d; }, 0.0f,
            thrust::plus<float>());
    thrust::transform(enumerate_begin(indices), enumerate_end(indices),
                      result.correspondence_set_.begin(),
                      [] __device__(const thrust::tuple<int, int> &idxs) {
                          int j = thrust::get<1>(idxs);
                          return (j < 0) ? Eigen::Vector2i(-1, -1)
                                         : Eigen::Vector2i(thrust::get<0>(idxs),
                                                           j);
                      });
    auto end =
            thrust::remove_if(result.correspondence_set_.begin(),
                              result.correspondence_set_.end(),
                              [] __device__(const Eigen::Vector2i &x) -> bool {
                                  return (x[0] < 0);
                              });
    int n_out = thrust::distance(result.correspondence_set_.begin(), end);
    result.correspondence_set_.resize(n_out);

    end = thrust::remove_if(result.correspondence_set_.begin(),
                            result.correspondence_set_.end(),
                            NormalRejector(thrust::raw_pointer_cast(source.normals_.data()),
                                           thrust::raw_pointer_cast(target.normals_.data()),
                                           max_normal_difference));
    
    n_out = thrust::distance(result.correspondence_set_.begin(), end);
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
    geometry::KDTreeFlann kdtree(target);
    geometry::PointCloud pcd = source;
    if (!transformation.isIdentity()) {
        pcd.Transform(transformation);
    }
    return GetRegistrationResultAndCorrespondences(
            pcd, target, kdtree, max_correspondence_distance, transformation);
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

RegistrationResult cupoch::registration::RegistrationICPWithNormalRejector(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        float max_correspondence_distance,
        float max_normal_difference,
        const Eigen::Matrix4f &init /* = Eigen::Matrix4f::Identity()*/,
        const TransformationEstimation &estimation
        /* = TransformationEstimationPointToPoint(false)*/,
        const ICPConvergenceCriteria
                &criteria /* = ICPConvergenceCriteria()*/) {

    if (max_correspondence_distance <= 0.0) {
      utility::LogError("Invalid max_correspondence_distance.");
    }
    
    if (!source.HasNormals() || !target.HasNormals()) {
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
    result = GetRegistrationResultAndCorrespondencesWithNormalRejection(
            pcd, target, kdtree, max_correspondence_distance,
            max_normal_difference, transformation);
    for (int i = 0; i < criteria.max_iteration_; i++) {
        utility::LogDebug("ICP Iteration #{:d}: Fitness {:.4f}, RMSE {:.4f}", i,
                          result.fitness_, result.inlier_rmse_);
        Eigen::Matrix4f update = estimation.ComputeTransformation(
                pcd, target, result.correspondence_set_);
        transformation = update * transformation;
        pcd.Transform(update);
        RegistrationResult backup = result;
        result = GetRegistrationResultAndCorrespondencesWithNormalRejection(
                pcd, target, kdtree, max_correspondence_distance,
                max_normal_difference, transformation);
        if (std::abs(backup.fitness_ - result.fitness_) <
                    criteria.relative_fitness_ &&
            std::abs(backup.inlier_rmse_ - result.inlier_rmse_) <
                    criteria.relative_rmse_) {
            break;
        }
    }
    return result;
}