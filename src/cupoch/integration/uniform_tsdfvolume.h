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

#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/integration/tsdfvolume.h"

namespace cupoch {

namespace geometry {

class TSDFVoxel {
public:
    __host__ __device__ TSDFVoxel() {}
    __host__ __device__ TSDFVoxel(const Eigen::Vector3f &color)
        : color_(color) {}
    __host__ __device__ ~TSDFVoxel() {}

public:
    float tsdf_ = 0;
    float weight_ = 0;
    Eigen::Vector3f color_ = Eigen::Vector3f(1.0, 1.0, 1.0);
};

}  // namespace geometry

namespace integration {

class UniformTSDFVolume : public TSDFVolume {
public:
    UniformTSDFVolume(float length,
                      int resolution,
                      float sdf_trunc,
                      TSDFVolumeColorType color_type,
                      const Eigen::Vector3f &origin = Eigen::Vector3f::Zero());
    ~UniformTSDFVolume() override;
    UniformTSDFVolume(const UniformTSDFVolume &other);

public:
    void Reset() override;
    void Integrate(const geometry::RGBDImage &image,
                   const camera::PinholeCameraIntrinsic &intrinsic,
                   const Eigen::Matrix4f &extrinsic) override;
    std::shared_ptr<geometry::PointCloud> ExtractPointCloud() override;
    std::shared_ptr<geometry::TriangleMesh> ExtractTriangleMesh() override;

    /// Debug function to extract the voxel data into a VoxelGrid
    std::shared_ptr<geometry::PointCloud> ExtractVoxelPointCloud() const;
    std::shared_ptr<geometry::VoxelGrid> ExtractVoxelGrid() const;

    /// Faster Integrate function that uses depth_to_camera_distance_multiplier
    /// precomputed from camera intrinsic
    void IntegrateWithDepthToCameraDistanceMultiplier(
            const geometry::RGBDImage &image,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4f &extrinsic,
            const geometry::Image &depth_to_camera_distance_multiplier);

    std::shared_ptr<geometry::PointCloud> Raycast(const camera::PinholeCameraIntrinsic &intrinsic,
                                                  const Eigen::Matrix4f &extrinsic,
                                                  float sdf_trunc,
                                                  bool project_valid_depth_only = true) const;

public:
    utility::device_vector<geometry::TSDFVoxel> voxels_;
    Eigen::Vector3f origin_;
    float length_;
    int resolution_;
    int voxel_num_;
};

}  // namespace integration
}  // namespace cupoch