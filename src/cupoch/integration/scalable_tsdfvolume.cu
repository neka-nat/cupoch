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
#include <stdgpu/unordered_map.cuh>

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/integration/scalable_tsdfvolume.h"
#include "cupoch/integration/marching_cubes_const.h"
#include "cupoch/integration/integrate_functor.h"
#include "cupoch/utility/console.h"

#include "cupoch/utility/range.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace integration {

typedef stdgpu::unordered_map<Eigen::Vector3i, ScalableTSDFVolume::VolumeUnit<>, utility::hash_eigen<Eigen::Vector3i>> VolumeUnitsMap;
class ScalableTSDFVolume::VolumeUnitsImpl {
public:
    VolumeUnitsMap volume_units_;
};

namespace {

struct scalable_integrate_functor : public integrate_functor {
    scalable_integrate_functor(float fx,
                               float fy,
                               float cx,
                               float cy,
                               const Eigen::Matrix4f &extrinsic,
                               float voxel_length,
                               float sdf_trunc,
                               float safe_width,
                               float safe_height,
                               int resolution,
                               const uint8_t *color,
                               const uint8_t *depth,
                               const uint8_t *depth_to_camera_distance_multiplier,
                               int width,
                               int num_of_channels,
                               TSDFVolumeColorType color_type,
                               VolumeUnitsMap volume_units)
          : integrate_functor(fx, fy, cx, cy,
                              extrinsic, voxel_length,
                              sdf_trunc, safe_width,
                              safe_height, resolution,
                              color, depth,
                              depth_to_camera_distance_multiplier,
                              width, num_of_channels,
                              color_type),
          volume_units_(volume_units) {};
    VolumeUnitsMap volume_units_;
    __device__ void operator() (size_t idx) {
        int res2 = resolution_ * resolution_;
        int res3 = res2 * resolution_;
        int n_v = idx / res3;
        int xyz = idx % res3;
        int x = xyz / res2;
        int yz = xyz % res2;
        int y = yz / resolution_;
        int z = yz % resolution_;
        auto& tsdfvol = (volume_units_.begin() + n_v)->second;
        if (tsdfvol.is_initialized_) {
            ComputeTSDF(tsdfvol.voxels_[xyz], tsdfvol.origin_, x, y, z);
        }
    }
};

__global__ void OpenVolumeUnitKernel(const Eigen::Vector3f* points,
                                     float sdf_trunc,
                                     float volume_unit_length,
                                     int n,
                                     VolumeUnitsMap volume_units) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > n) return;
    auto min_bound = LocateVolumeUnit(points[idx] - Eigen::Vector3f::Constant(sdf_trunc), volume_unit_length);
    auto max_bound = LocateVolumeUnit(points[idx] + Eigen::Vector3f::Constant(sdf_trunc), volume_unit_length);
    for (auto x = min_bound(0); x <= max_bound(0); x++) {
        for (auto y = min_bound(1); y <= max_bound(1); y++) {
            for (auto z = min_bound(2); z <= max_bound(2); z++) {
                Eigen::Vector3i loc = Eigen::Vector3i(x, y, z);
                if (!volume_units.contains(loc)) {
                    volume_units.emplace(loc, ScalableTSDFVolume::VolumeUnit<>(loc.cast<float>() * volume_unit_length));
                }
            }
        }
    }
}

struct extract_pointcloud_functor {
    extract_pointcloud_functor(const VolumeUnitsMap& volume_units,
                               const stdgpu::device_indexed_range<const VolumeUnitsMap::value_type>& range,
                               int resolution,
                               float voxel_length,
                               float volume_unit_length,
                               TSDFVolumeColorType color_type,
                               Eigen::Vector3f* points,
                               Eigen::Vector3f* normals,
                               Eigen::Vector3f* colors)
        : volume_units_(volume_units),
          range_(range),
          resolution_(resolution),
          voxel_length_(voxel_length),
          half_voxel_length_(0.5 * voxel_length_),
          volume_unit_length_(volume_unit_length),
          color_type_(color_type),
          points_(points),
          normals_(normals),
          colors_(colors) {};
    const VolumeUnitsMap volume_units_;
    const stdgpu::device_indexed_range<const VolumeUnitsMap::value_type> range_;
    const int resolution_;
    const float voxel_length_;
    const float half_voxel_length_;
    const float volume_unit_length_;
    const TSDFVolumeColorType color_type_;
    Eigen::Vector3f* points_;
    Eigen::Vector3f* normals_;
    Eigen::Vector3f* colors_;
    __device__ Eigen::Vector3f GetNormalAt(const Eigen::Vector3f &p) {
        Eigen::Vector3f n;
        const float half_gap = 0.99 * voxel_length_;
        for (int i = 0; i < 3; i++) {
            Eigen::Vector3f p0 = p;
            p0(i) -= half_gap;
            Eigen::Vector3f p1 = p;
            p1(i) += half_gap;
            n(i) = GetTSDFAt(p1) - GetTSDFAt(p0);
        }
        return n.normalized();
    }
    __device__ float GetTSDFAt(const Eigen::Vector3f &p) {
        Eigen::Vector3f p_locate =
                p - Eigen::Vector3f(0.5, 0.5, 0.5) * voxel_length_;
        Eigen::Vector3i index0 = LocateVolumeUnit(p_locate, volume_unit_length_);
        auto unit_itr = volume_units_.find(index0);
        if (unit_itr == volume_units_.end()) {
            return 0.0;
        }
        const auto &volume0 = unit_itr->second;
        Eigen::Vector3i idx0;
        Eigen::Vector3f p_grid =
                (p_locate - index0.cast<float>() * volume_unit_length_) /
                voxel_length_;
        for (int i = 0; i < 3; i++) {
            idx0(i) = (int)floorf(p_grid(i));
            if (idx0(i) < 0) idx0(i) = 0;
            if (idx0(i) >= resolution_)
                idx0(i) = resolution_ - 1;
        }
        Eigen::Vector3f r = p_grid - idx0.cast<float>();
        float f[8];
        for (int i = 0; i < 8; i++) {
            Eigen::Vector3i index1 = index0;
            Eigen::Vector3i idx1 = idx0 + Eigen::Vector3i(shift[i][0], shift[i][1], shift[i][2]);
            if (idx1(0) < resolution_ &&
                idx1(1) < resolution_ &&
                idx1(2) < resolution_) {
                f[i] = volume0.voxels_[IndexOf(idx1, resolution_)].tsdf_;
            } else {
                for (int j = 0; j < 3; j++) {
                    if (idx1(j) >= resolution_) {
                        idx1(j) -= resolution_;
                        index1(j) += 1;
                    }
                }
                auto unit_itr1 = volume_units_.find(index1);
                if (unit_itr1 == volume_units_.end()) {
                    f[i] = 0.0f;
                } else {
                    const auto &volume1 = unit_itr1->second;
                    f[i] = volume1.voxels_[IndexOf(idx1, resolution_)].tsdf_;
                }
            }
        }
        return (1 - r(0)) * ((1 - r(1)) * ((1 - r(2)) * f[0] + r(2) * f[4]) +
                             r(1) * ((1 - r(2)) * f[3] + r(2) * f[7])) +
               r(0) * ((1 - r(1)) * ((1 - r(2)) * f[1] + r(2) * f[5]) +
                       r(1) * ((1 - r(2)) * f[2] + r(2) * f[6]));
    }
    __device__ void operator()(const size_t idx) {
        int res2 = resolution_ * resolution_;
        int res3 = res2 * resolution_;
        int n_v = idx / res3;
        int xyz = idx % res3;
        int x = xyz / res2;
        int yz = xyz % res2;
        int y = yz / resolution_;
        int z = yz % resolution_;
        const auto pair_val = *(range_.begin() + n_v);
        const auto& index0 = pair_val.first;
        const auto& volume0 = pair_val.second;
        Eigen::Vector3i idx0(x, y, z);
        float w0 = volume0.voxels_[IndexOf(idx0, resolution_)].weight_;
        float f0 = volume0.voxels_[IndexOf(idx0, resolution_)].tsdf_;
        Eigen::Vector3f c0 = Eigen::Vector3f::Zero();
        if (color_type_ != TSDFVolumeColorType::NoColor)
            c0 = volume0.voxels_[IndexOf(idx0, resolution_)].color_;
        if (w0 != 0.0f && f0 < 0.98f && f0 >= -0.98f) {
            Eigen::Vector3f p0 =
                    Eigen::Vector3f(half_voxel_length_ +
                                            voxel_length_ * x,
                                    half_voxel_length_ +
                                            voxel_length_ * y,
                                    half_voxel_length_ +
                                            voxel_length_ * z) +
                    index0.cast<float>() * volume_unit_length_;
            float w1, f1;
            Eigen::Vector3f c1;
            for (int i = 0; i < 3; i++) {
                Eigen::Vector3f p1 = p0;
                Eigen::Vector3i idx1 = idx0;
                Eigen::Vector3i index1 = index0;
                p1(i) += voxel_length_;
                idx1(i) += 1;
                if (idx1(i) < resolution_) {
                    w1 = volume0.voxels_[IndexOf(idx1, resolution_)].weight_;
                    f1 = volume0.voxels_[IndexOf(idx1, resolution_)].tsdf_;
                    if (color_type_ !=
                        TSDFVolumeColorType::NoColor)
                        c1 = volume0.voxels_[IndexOf(idx1, resolution_)].color_;
                } else {
                    idx1(i) -= resolution_;
                    index1(i) += 1;
                    auto unit_itr = volume_units_.find(index1);
                    if (unit_itr == volume_units_.end()) {
                        w1 = 0.0f;
                        f1 = 0.0f;
                    } else {
                        const auto &volume1 = unit_itr->second;
                        w1 = volume1.voxels_[IndexOf(idx1, resolution_)].weight_;
                        f1 = volume1.voxels_[IndexOf(idx1, resolution_)].tsdf_;
                        if (color_type_ !=
                            TSDFVolumeColorType::NoColor)
                            c1 = volume1.voxels_[IndexOf(idx1, resolution_)].color_;
                    }
                }
                if (w1 != 0.0f && f1 < 0.98f && f1 >= -0.98f &&
                    f0 * f1 < 0) {
                    float r0 = abs(f0);
                    float r1 = abs(f1);
                    Eigen::Vector3f p = p0;
                    p(i) = (p0(i) * r1 + p1(i) * r0) /
                           (r0 + r1);
                    points_[idx * 3 + i] = p;
                    if (color_type_ ==
                        TSDFVolumeColorType::RGB8) {
                        colors_[idx * 3 + i] = 
                                ((c0 * r1 + c1 * r0) /
                                 (r0 + r1) / 255.0f);
                    } else if (color_type_ ==
                               TSDFVolumeColorType::Gray32) {
                        colors_[idx * 3 + i] = 
                                ((c0 * r1 + c1 * r0) /
                                 (r0 + r1));
                    }
                    // has_normal
                    normals_[idx * 3 + i] = GetNormalAt(p);
                }
            }
        }
    }
};

}

ScalableTSDFVolume::ScalableTSDFVolume(float voxel_length,
                                       float sdf_trunc,
                                       TSDFVolumeColorType color_type,
                                       int depth_sampling_stride /* = 4*/,
                                       int map_size /* = 1000*/)
    : TSDFVolume(voxel_length, sdf_trunc, color_type),
      volume_unit_length_(voxel_length * VolumeUnit<>::GetResolution()),
      resolution_(VolumeUnit<>::GetResolution()),
      volume_unit_voxel_num_(VolumeUnit<>::GetVoxelNum()),
      depth_sampling_stride_(depth_sampling_stride) {
    impl_ = std::make_shared<VolumeUnitsImpl>();
    impl_->volume_units_ = VolumeUnitsMap::createDeviceObject(map_size);
}

ScalableTSDFVolume::~ScalableTSDFVolume() {
    VolumeUnitsMap::destroyDeviceObject(impl_->volume_units_);
}

void ScalableTSDFVolume::Reset() {
    impl_->volume_units_.clear();
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
                                              sdf_trunc_, volume_unit_length_,
                                              n_points, impl_->volume_units_);
    cudaSafeCall(cudaDeviceSynchronize());
    IntegrateWithDepthToCameraDistanceMultiplier(image, intrinsic, extrinsic,
                                                 *depth2cameradistance);
}


std::shared_ptr<geometry::PointCloud> ScalableTSDFVolume::ExtractPointCloud() {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    size_t n_total = impl_->volume_units_.size() * volume_unit_voxel_num_;
    const Eigen::Vector3f nanvec = Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
    pointcloud->points_.resize(3 * n_total, nanvec);
    pointcloud->normals_.resize(3 * n_total, nanvec);
    pointcloud->colors_.resize(3 * n_total, nanvec);
    extract_pointcloud_functor func(impl_->volume_units_,
                                    impl_->volume_units_.device_range(),
                                    resolution_,
                                    voxel_length_,
                                    volume_unit_length_,
                                    color_type_,
                                    thrust::raw_pointer_cast(pointcloud->points_.data()),
                                    thrust::raw_pointer_cast(pointcloud->normals_.data()),
                                    thrust::raw_pointer_cast(pointcloud->colors_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(n_total), func);
    pointcloud->RemoveNoneFinitePoints(true, true);
    return pointcloud;
}

std::shared_ptr<geometry::TriangleMesh> ScalableTSDFVolume::ExtractTriangleMesh() {
    utility::LogError("ScalableTSDFVolume::ExtractTriangleMesh is not impelemented");
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    return mesh;
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
            impl_->volume_units_);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(
                             impl_->volume_units_.max_size() * VolumeUnit<>::GetVoxelNum()),
                     func);
}

}  // namespace integration
}  // namespace cupoch