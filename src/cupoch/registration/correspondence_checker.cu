/**
 * Copyright (c) 2024 Neka-Nat
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
#include "cupoch/registration/correspondence_checker.h"

#include <Eigen/Dense>
#include <thrust/logical.h>

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace registration {

namespace {
    struct edge_length_checker_functor {
        float similarity_threshold_;
        size_t corres_size_;
        const Eigen::Vector3f* source_points_;
        const Eigen::Vector3f* target_points_;
        const Eigen::Vector2i* corres_;
        edge_length_checker_functor(
            float similarity_threshold,
            size_t corres_size,
            const Eigen::Vector3f* source_points,
            const Eigen::Vector3f* target_points,
            const Eigen::Vector2i* corres)
            : similarity_threshold_(similarity_threshold),
              corres_size_(corres_size),
              source_points_(source_points),
              target_points_(target_points),
              corres_(corres) {}
        __device__ bool operator()(int idx) {
            int i = idx / corres_size_;
            int j = idx % corres_size_;
            if (i == j) return true;
             float dis_source = (source_points_[corres_[i](0)] - source_points_[corres_[j](0)]).norm();
            float dis_target = (target_points_[corres_[i](1)] - target_points_[corres_[j](1)]).norm();
            return dis_source >= dis_target * similarity_threshold_ &&
                   dis_target >= dis_source * similarity_threshold_;
        }

    };

    struct distance_checker_functor {
        float distance_threshold_;
        const Eigen::Vector3f* source_points_;
        const Eigen::Vector3f* target_points_;
        const Eigen::Matrix4f transformation_;
        distance_checker_functor(
            float distance_threshold,
            const Eigen::Vector3f* source_points,
            const Eigen::Vector3f* target_points,
            const Eigen::Matrix4f transformation)
            : distance_threshold_(distance_threshold),
              source_points_(source_points),
              target_points_(target_points),
              transformation_(transformation) {}
        __device__ bool operator()(const Eigen::Vector2i& corr) {
            const auto &pt = source_points_[corr(0)];
            Eigen::Vector3f pt_trans =
                    (transformation_ * Eigen::Vector4f(pt(0), pt(1), pt(2), 1.0))
                            .block<3, 1>(0, 0);
            return (target_points_[
                corr(1)] - pt_trans).norm() <= distance_threshold_;
        }
    };

    struct normal_checker_functor {
        float cos_normal_angle_threshold_;
        const Eigen::Vector3f* source_points_;
        const Eigen::Vector3f* target_points_;
        const Eigen::Vector3f* source_normals_;
        const Eigen::Vector3f* target_normals_;
        const Eigen::Matrix4f transformation_;
        normal_checker_functor(
            float cos_normal_angle_threshold,
            const Eigen::Vector3f* source_points,
            const Eigen::Vector3f* target_points,
            const Eigen::Vector3f* source_normals,
            const Eigen::Vector3f* target_normals,
            const Eigen::Matrix4f transformation)
            : cos_normal_angle_threshold_(cos_normal_angle_threshold),
              source_points_(source_points),
              target_points_(target_points),
              source_normals_(source_normals),
              target_normals_(target_normals),
              transformation_(transformation) {}
        __device__ bool operator()(const Eigen::Vector2i& corr) {
            const auto &normal = source_normals_[corr(0)];
            Eigen::Vector3f normal_trans =
                    (transformation_ *
                     Eigen::Vector4f(normal(0), normal(1), normal(2), 0.0))
                            .block<3, 1>(0, 0);
            return target_normals_[corr(1)].dot(normal) >= cos_normal_angle_threshold_;
        }
    };
}

bool CorrespondenceCheckerBasedOnEdgeLength::Check(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4f & /*transformation*/) const {
    return thrust::all_of(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator(corres.size() * corres.size()),
        edge_length_checker_functor(
            similarity_threshold_,
            corres.size(),
            thrust::raw_pointer_cast(source.points_.data()),
            thrust::raw_pointer_cast(target.points_.data()),
            thrust::raw_pointer_cast(corres.data())));
}

bool CorrespondenceCheckerBasedOnDistance::Check(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4f &transformation) const {
    return thrust::all_of(
        corres.begin(), corres.end(),
        distance_checker_functor(
            distance_threshold_,
            thrust::raw_pointer_cast(source.points_.data()),
            thrust::raw_pointer_cast(target.points_.data()),
            transformation));
}

bool CorrespondenceCheckerBasedOnNormal::Check(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres,
        const Eigen::Matrix4f &transformation) const {
    if (!source.HasNormals() || !target.HasNormals()) {
        utility::LogWarning(
                "[CorrespondenceCheckerBasedOnNormal::Check] Pointcloud has no "
                "normals.");
        return true;
    }
    float cos_normal_angle_threshold = std::cos(normal_angle_threshold_);
    return thrust::all_of(
        corres.begin(), corres.end(),
        normal_checker_functor(
            cos_normal_angle_threshold,
            thrust::raw_pointer_cast(source.points_.data()),
            thrust::raw_pointer_cast(target.points_.data()),
            thrust::raw_pointer_cast(source.normals_.data()),
            thrust::raw_pointer_cast(target.normals_.data()),
            transformation));
}

}  // namespace registration
}  // namespace cupoch
