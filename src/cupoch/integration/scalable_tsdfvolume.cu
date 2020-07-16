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
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/integration/scalable_tsdfvolume.h"
#include "cupoch/integration/integrate_functor.h"
#include "cupoch/utility/console.h"

#include "cupoch/utility/range.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace integration {

namespace {

__global__ void OpenVolumeUnitKernel(const Eigen::Vector3f* points,
                                     float sdf_trunc,
                                     float volume_unit_length,
                                     float voxel_length,
                                     int n,
                                     ScalableTSDFVolume::VolumeUnitsMap volume_units) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n) return;
    auto min_bound = LocateVolumeUnit(points[idx] - Eigen::Vector3f::Constant(sdf_trunc), volume_unit_length);
    auto max_bound = LocateVolumeUnit(points[idx] + Eigen::Vector3f::Constant(sdf_trunc), volume_unit_length);
    for (auto x = min_bound(0); x <= max_bound(0); x++) {
        for (auto y = min_bound(1); y <= max_bound(1); y++) {
            for (auto z = min_bound(2); z <= max_bound(2); z++) {
                    auto loc = Eigen::Vector3i(x, y, z);
                    if (!volume_units.contains(loc)) {
                        volume_units.emplace(loc, ScalableTSDFVolume::VolumeUnit<>(loc.cast<float>() * volume_unit_length));
                    }
            }
        }
    }
}

}

ScalableTSDFVolume::ScalableTSDFVolume(float voxel_length,
                                       float sdf_trunc,
                                       TSDFVolumeColorType color_type,
                                       int depth_sampling_stride /* = 4*/,
                                       int map_size /* = 10000*/)
    : TSDFVolume(voxel_length, sdf_trunc, color_type),
      volume_unit_length_(voxel_length * VolumeUnit<>::GetResolution()),
      resolution_(VolumeUnit<>::GetResolution()),
      volume_unit_voxel_num_(VolumeUnit<>::GetVoxelNum()),
      depth_sampling_stride_(depth_sampling_stride) {
    volume_units_ = VolumeUnitsMap::createDeviceObject(map_size);
}

ScalableTSDFVolume::~ScalableTSDFVolume() {
    VolumeUnitsMap::destroyDeviceObject(volume_units_);
}

void ScalableTSDFVolume::Reset() {
    volume_units_.clear();
}

void ScalableTSDFVolume::Integrate(
        const geometry::RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic) {
    if ((image.depth_.num_of_channels_ != 1) ||
        (image.depth_.bytes_per_channel_ != 4) ||
        (image.depth_.width_ != intrinsic.width_) ||
        (image.depth_.height_ != intrinsic.height_) ||
        (color_type_ == TSDFVolumeColorType::RGB8 &&
         image.color_.num_of_channels_ != 3) ||
        (color_type_ == TSDFVolumeColorType::RGB8 &&
         image.color_.bytes_per_channel_ != 1) ||
        (color_type_ == TSDFVolumeColorType::Gray32 &&
         image.color_.num_of_channels_ != 1) ||
        (color_type_ == TSDFVolumeColorType::Gray32 &&
         image.color_.bytes_per_channel_ != 4) ||
        (color_type_ != TSDFVolumeColorType::NoColor &&
         image.color_.width_ != intrinsic.width_) ||
        (color_type_ != TSDFVolumeColorType::NoColor &&
         image.color_.height_ != intrinsic.height_)) {
        utility::LogError(
                "[ScalableTSDFVolume::Integrate] Unsupported image format.");
    }
    auto depth2cameradistance =
            geometry::Image::CreateDepthToCameraDistanceMultiplierFloatImage(
                    intrinsic);
    auto pointcloud = geometry::PointCloud::CreateFromDepthImage(
            image.depth_, intrinsic, extrinsic, 1000.0, 1000.0,
            depth_sampling_stride_);
    size_t n_points = pointcloud->points_.size();
    const dim3 threads(32);
    const dim3 blocks((n_points + threads.x - 1) / threads.x);
    OpenVolumeUnitKernel<<<blocks, threads>>>(thrust::raw_pointer_cast(pointcloud->points_.data()),
                                              sdf_trunc_, volume_unit_length_, voxel_length_,
                                              n_points, volume_units_);
    cudaSafeCall(cudaDeviceSynchronize());
    IntegrateWithDepthToCameraDistanceMultiplier(image, intrinsic, extrinsic,
                                                 *depth2cameradistance);
}

void ScalableTSDFVolume::IntegrateWithDepthToCameraDistanceMultiplier(
        const geometry::RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic,
        const geometry::Image &depth_to_camera_distance_multiplier) {
    const float fx = intrinsic.GetFocalLength().first;
    const float fy = intrinsic.GetFocalLength().second;
    const float cx = intrinsic.GetPrincipalPoint().first;
    const float cy = intrinsic.GetPrincipalPoint().second;
    const float safe_width = intrinsic.width_ - 0.0001f;
    const float safe_height = intrinsic.height_ - 0.0001f;
    scalable_integrate_functor func(
            fx, fy, cx, cy, extrinsic, voxel_length_, sdf_trunc_,
            safe_width, safe_height, resolution_,
            thrust::raw_pointer_cast(image.color_.data_.data()),
            thrust::raw_pointer_cast(image.depth_.data_.data()),
            thrust::raw_pointer_cast(
                    depth_to_camera_distance_multiplier.data_.data()),
            image.depth_.width_, image.color_.num_of_channels_, color_type_,
            VolumeUnit<>::GetVoxelNum(), volume_units_);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(
                             volume_units_.size() * VolumeUnit<>::GetVoxelNum()),
                     func);
}

}  // namespace integration
}  // namespace cupoch