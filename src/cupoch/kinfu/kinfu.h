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
#pragma once
#include <vector>

#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "cupoch/geometry/rgbdimage.h"
#include "cupoch/integration/uniform_tsdfvolume.h"
#include "cupoch/registration/transformation_estimation.h"

namespace cupoch {

// https://github.com/chrdiller/KinectFusionLib
namespace kinfu {

typedef std::vector<std::shared_ptr<geometry::PointCloud>> PointCloudPyramid;

class KinfuOption {
public:
    KinfuOption(
            int num_pyramid_levels = 4,
            int diameter = 1,
            float sigma_depth = 0.1f,
            float sigma_space = 5.0f,
            float depth_cutoff = 3.0f,
            float tsdf_length = 8.0f,
            int tsdf_resolution = 512,
            float sdf_trunc = 0.05f,
            integration::TSDFVolumeColorType tsdf_color_type =
                    integration::TSDFVolumeColorType::RGB8,
            const Eigen::Vector3f& tsdf_origin = Eigen::Vector3f::Zero(),
            float distance_threshold = 0.3f,
            const std::vector<int>& icp_iterations = {20, 20, 20, 20},
            registration::TransformationEstimationType tf_type = registration::TransformationEstimationType::PointToPlane)
        : num_pyramid_levels_(num_pyramid_levels),
          diameter_(diameter),
          sigma_depth_(sigma_depth),
          sigma_space_(sigma_space),
          depth_cutoff_(depth_cutoff),
          tsdf_length_(tsdf_length),
          tsdf_resolution_(tsdf_resolution),
          sdf_trunc_(sdf_trunc),
          tsdf_color_type_(tsdf_color_type),
          tsdf_origin_(tsdf_origin),
          distance_threshold_(distance_threshold),
          icp_iterations_(icp_iterations),
          tf_type_(tf_type) {};
    ~KinfuOption(){};
    int num_pyramid_levels_;
    int diameter_;
    float sigma_depth_;
    float sigma_space_;
    float depth_cutoff_;
    float tsdf_length_;
    int tsdf_resolution_;
    float sdf_trunc_;
    integration::TSDFVolumeColorType tsdf_color_type_;
    Eigen::Vector3f tsdf_origin_;
    float distance_threshold_;
    std::vector<int> icp_iterations_;
    registration::TransformationEstimationType tf_type_;
};

class KinfuPipeline {
public:
    KinfuPipeline(const camera::PinholeCameraIntrinsic& intrinsic,
             const KinfuOption& option = KinfuOption());
    ~KinfuPipeline();

    void Reset();
    bool ProcessFrame(const geometry::RGBDImage& image);
    std::shared_ptr<geometry::PointCloud> ExtractPointCloud();
    std::shared_ptr<geometry::TriangleMesh> ExtractTriangleMesh();

private:
    std::tuple<geometry::RGBDImagePyramid,
               geometry::RGBDImagePyramid,
               PointCloudPyramid>
    SurfaceMeasurement(const geometry::RGBDImage& image) const;

    std::tuple<Eigen::Matrix4f, bool> PoseEstimation(
            const Eigen::Matrix4f& extrinsic,
            const PointCloudPyramid& frame_data,
            const PointCloudPyramid& target_data);

public:
    camera::PinholeCameraIntrinsic intrinsic_;
    Eigen::Matrix4f cur_pose_ = Eigen::Matrix4f::Identity();
    int frame_id_ = 0;
    integration::UniformTSDFVolume volume_;
    PointCloudPyramid model_pyramid_;
    KinfuOption option_;
};

}  // namespace kinfu
}  // namespace cupoch