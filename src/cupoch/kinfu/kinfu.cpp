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
#include "cupoch/kinfu/kinfu.h"

#include "cupoch/registration/registration.h"
#include "cupoch/registration/colored_icp.h"

namespace cupoch {
namespace kinfu {

KinfuPipeline::KinfuPipeline(const camera::PinholeCameraIntrinsic& intrinsic,
                   const KinfuOption& option)
    : intrinsic_(intrinsic),
      volume_(option.tsdf_length_,
              option.tsdf_resolution_,
              option.sdf_trunc_,
              option.tsdf_color_type_,
              option.tsdf_origin_),
      model_pyramid_(option.num_pyramid_levels_),
      option_(option) {}

KinfuPipeline::~KinfuPipeline() {}

void KinfuPipeline::Reset() {
    cur_pose_ = Eigen::Matrix4f::Identity();
    volume_.Reset();
    for (auto m : model_pyramid_) {
        m.reset();
    }
    frame_id_ = 0;
}

bool KinfuPipeline::ProcessFrame(const geometry::RGBDImage& image) {
    if (!image.color_.HasData() || !image.depth_.HasData()) {
        return false;
    }
    geometry::RGBDImagePyramid img_pyramid, smooth_img_pyramid;
    Eigen::Matrix4f extrinsic = utility::InverseTransform(cur_pose_);
    PointCloudPyramid pc_pyramid;
    std::tie(img_pyramid, smooth_img_pyramid, pc_pyramid) =
            SurfaceMeasurement(image);
    if (frame_id_ > 0) {
        bool icp_success = true;
        std::tie(extrinsic, icp_success) =
                PoseEstimation(extrinsic, model_pyramid_, pc_pyramid);
        if (!icp_success) {
            return false;
        }
        cur_pose_ = utility::InverseTransform(extrinsic);
    }
    volume_.Integrate(*smooth_img_pyramid[0], intrinsic_, extrinsic);
    for (int i = 0; i < option_.num_pyramid_levels_; ++i) {
        model_pyramid_[i] = volume_.Raycast(intrinsic_.CreatePyramidLevel(i),
                                            extrinsic, option_.sdf_trunc_);
    }
    frame_id_++;
    return true;
}

std::shared_ptr<geometry::PointCloud> KinfuPipeline::ExtractPointCloud() {
    return volume_.ExtractPointCloud();
}

std::shared_ptr<geometry::TriangleMesh> KinfuPipeline::ExtractTriangleMesh() {
    return volume_.ExtractTriangleMesh();
}

std::tuple<geometry::RGBDImagePyramid,
           geometry::RGBDImagePyramid,
           PointCloudPyramid>
KinfuPipeline::SurfaceMeasurement(const geometry::RGBDImage& image) const {
    auto img_pyramid = image.CreatePyramid(option_.num_pyramid_levels_);
    auto smooth_img_pyramid = geometry::RGBDImage::BilateralFilterPyramidDepth(
            img_pyramid, option_.diameter_, option_.sigma_depth_,
            option_.sigma_space_);
    PointCloudPyramid pc_pyramid(option_.num_pyramid_levels_);
    for (int i = 0; i < option_.num_pyramid_levels_; ++i) {
        pc_pyramid[i] = geometry::PointCloud::CreateFromRGBDImage(
                *smooth_img_pyramid[i], intrinsic_, Eigen::Matrix4f::Identity(),
                true, option_.depth_cutoff_, true);
    }
    return std::make_tuple(std::move(img_pyramid),
                           std::move(smooth_img_pyramid),
                           std::move(pc_pyramid));
}

std::tuple<Eigen::Matrix4f, bool> KinfuPipeline::PoseEstimation(
        const Eigen::Matrix4f& extrinsic,
        const PointCloudPyramid& frame_data,
        const PointCloudPyramid& target_data) {
    Eigen::Matrix4f cur_global_trans = extrinsic;

    for (int level = option_.num_pyramid_levels_ - 1; level >= 0; --level) {
        registration::ICPConvergenceCriteria criteria;
        criteria.max_iteration_ = option_.icp_iterations_[level];
        switch (option_.tf_type_) {
            case registration::TransformationEstimationType::PointToPlane: {
                auto res = registration::RegistrationICP(
                        *frame_data[level], *target_data[level],
                        option_.distance_threshold_, cur_global_trans,
                        registration::TransformationEstimationPointToPlane(100000),
                        criteria);
                cur_global_trans = res.transformation_;
                break;
            }
            case registration::TransformationEstimationType::ColoredICP: {
                auto res = registration::RegistrationColoredICP(
                        *frame_data[level], *target_data[level],
                        option_.distance_threshold_, cur_global_trans,
                        criteria, 0.968, 100000);
                cur_global_trans = res.transformation_;
                break;
            }
            default: {
                utility::LogError(
                       "[KinfuPipeline::PoseEstimation] Unsupported transformation type.");
                break;
            }
        }
    }
    return std::make_tuple(cur_global_trans, true);
}

}  // namespace kinfu
}  // namespace cupoch