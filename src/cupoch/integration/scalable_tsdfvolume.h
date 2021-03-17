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

#include <memory>

#include "cupoch/integration/uniform_tsdfvolume.h"
#include "cupoch/utility/helper.h"

namespace cupoch {
namespace integration {

class UniformTSDFVolume;

/// The ScalableTSDFVolume implements a more memory efficient data structure for
/// volumetric integration.
///
/// This implementation is based on the following repository:
/// https://github.com/qianyizh/ElasticReconstruction/tree/master/Integrate
/// The reference is:
/// Q.-Y. Zhou and V. Koltun
/// Dense Scene Reconstruction with Points of Interest
/// In SIGGRAPH 2013
///
/// An observed depth pixel gives two types of information: (a) an approximation
/// of the nearby surface, and (b) empty space from the camera to the surface.
/// They induce two core concepts of volumetric integration: weighted average of
/// a truncated signed distance function (TSDF), and carving. The weighted
/// average of TSDF is great in addressing the Gaussian noise along surface
/// normal and producing a smooth surface output. The carving is great in
/// removing outlier structures like floating noise pixels and bumps along
/// structure edges.
class ScalableTSDFVolume : public TSDFVolume {
public:
    template <int Num = 16>
    struct VolumeUnit {
    public:
        __host__ __device__ VolumeUnit(const Eigen::Vector3f& origin) : origin_(origin), is_initialized_(true) {};
        __host__ __device__ ~VolumeUnit() {};

    public:
        geometry::TSDFVoxel voxels_[Num * Num * Num];
        Eigen::Vector3f origin_;
        const int voxel_num_ = Num * Num * Num;
        bool is_initialized_ = false;
        static int GetResolution() { return Num; };
        static int GetVoxelNum() { return Num * Num * Num; };
    };

public:
    ScalableTSDFVolume(float voxel_length,
                       float sdf_trunc,
                       TSDFVolumeColorType color_type,
                       int depth_sampling_stride = 4,
                       int map_size = 1000);
    ~ScalableTSDFVolume() override;

public:
    void Reset() override;
    void Integrate(const geometry::RGBDImage &image,
                   const camera::PinholeCameraIntrinsic &intrinsic,
                   const Eigen::Matrix4f &extrinsic) override;
    std::shared_ptr<geometry::PointCloud> ExtractPointCloud() override;
    std::shared_ptr<geometry::TriangleMesh> ExtractTriangleMesh() override;
    /// Debug function to extract the voxel data into a point cloud.
    std::shared_ptr<geometry::PointCloud> ExtractVoxelPointCloud();

    void IntegrateWithDepthToCameraDistanceMultiplier(
            const geometry::RGBDImage &image,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4f &extrinsic,
            const geometry::Image &depth_to_camera_distance_multiplier);
public:
    /// Assume the index of the volume key is (x, y, z), then the unit spans
    /// from (x, y, z) * volume_unit_length_
    /// to (x + 1, y + 1, z + 1) * volume_unit_length_
    class VolumeUnitsImpl;
    std::shared_ptr<VolumeUnitsImpl> impl_;
    float volume_unit_length_;
    int resolution_;
    int volume_unit_voxel_num_;
    int depth_sampling_stride_;
};

__device__
inline Eigen::Vector3i LocateVolumeUnit(const Eigen::Vector3f &point,
                                        float volume_unit_length) {
    return Eigen::Vector3i((int)floorf(point(0) / volume_unit_length),
                           (int)floorf(point(1) / volume_unit_length),
                           (int)floorf(point(2) / volume_unit_length));
}

}  // namespace integration
}  // namespace cupoch