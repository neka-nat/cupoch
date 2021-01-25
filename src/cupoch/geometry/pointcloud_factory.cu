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
#include <Eigen/Dense>
#include <limits>
#include <thrust/iterator/discard_iterator.h>

#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "cupoch/geometry/image.h"
#include "cupoch/geometry/laserscanbuffer.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/rgbdimage.h"
#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/range.h"
#include "cupoch/utility/helper.h"

namespace cupoch {
namespace geometry {

namespace {

struct depth_to_pointcloud_functor {
    depth_to_pointcloud_functor(
            const uint8_t *depth,
            const int width,
            const int stride,
            const thrust::pair<float, float> &principal_point,
            const thrust::pair<float, float> &focal_length,
            const Eigen::Matrix4f &camera_pose)
        : depth_(depth),
          width_(width),
          stride_(stride),
          principal_point_(principal_point),
          focal_length_(focal_length),
          camera_pose_(camera_pose){};
    const uint8_t *depth_;
    const int width_;
    const int stride_;
    const thrust::pair<float, float> principal_point_;
    const thrust::pair<float, float> focal_length_;
    const Eigen::Matrix4f camera_pose_;
    __device__ Eigen::Vector3f operator()(size_t idx) {
        int strided_width = width_ / stride_;
        int row = idx / strided_width * stride_;
        int col = idx % strided_width * stride_;
        const float d = *(float *)(&depth_[(row * width_ + col) * sizeof(float)]);
        if (d <= 0.0) {
            return Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
        } else {
            float z = d;
            float x = (col - principal_point_.first) * z / focal_length_.first;
            float y =
                    (row - principal_point_.second) * z / focal_length_.second;
            Eigen::Vector4f point =
                    camera_pose_ * Eigen::Vector4f(x, y, z, 1.0);
            return point.block<3, 1>(0, 0);
        }
    }
};

std::shared_ptr<PointCloud> CreatePointCloudFromFloatDepthImage(
        const Image &depth,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic,
        int stride) {
    auto pointcloud = std::make_shared<PointCloud>();
    const Eigen::Matrix4f camera_pose = extrinsic.inverse();
    const auto focal_length = intrinsic.GetFocalLength();
    const auto principal_point = intrinsic.GetPrincipalPoint();
    const size_t depth_size = (depth.width_ / stride) * (depth.height_ / stride);
    pointcloud->points_.resize(depth_size);
    depth_to_pointcloud_functor func(
            thrust::raw_pointer_cast(depth.data_.data()), depth.width_, stride,
            principal_point, focal_length, camera_pose);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(depth_size),
                      pointcloud->points_.begin(), func);
    pointcloud->RemoveNoneFinitePoints(true, true);
    return pointcloud;
}

template <typename TC, int NC>
struct convert_from_rgbdimage_functor {
    convert_from_rgbdimage_functor(
            const uint8_t *depth,
            const uint8_t *color,
            int width,
            const Eigen::Matrix4f &camera_pose,
            const thrust::pair<float, float> &principal_point,
            const thrust::pair<float, float> &focal_length,
            float scale,
            float depth_cutoff)
        : depth_(depth),
          color_(color),
          width_(width),
          camera_pose_(camera_pose),
          principal_point_(principal_point),
          focal_length_(focal_length),
          scale_(scale),
          depth_cutoff_(depth_cutoff) {};
    const uint8_t *depth_;
    const uint8_t *color_;
    const int width_;
    const Eigen::Matrix4f camera_pose_;
    const thrust::pair<float, float> principal_point_;
    const thrust::pair<float, float> focal_length_;
    const float scale_;
    const float depth_cutoff_;
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> operator()(
            size_t idx) const {
        int i = idx / width_;
        int j = idx % width_;
        float *p = (float *)(depth_ + idx * sizeof(float));
        TC *pc = (TC *)(color_ + idx * NC * sizeof(TC));
        if (*p > 0 && (depth_cutoff_ <= 0 || depth_cutoff_ > *p)) {
            float z = (float)(*p);
            float x = (j - principal_point_.first) * z / focal_length_.first;
            float y = (i - principal_point_.second) * z / focal_length_.second;
            Eigen::Vector4f point =
                    camera_pose_ * Eigen::Vector4f(x, y, z, 1.0);
            Eigen::Vector3f points = point.block<3, 1>(0, 0);
            Eigen::Vector3f colors =
                    Eigen::Vector3f(pc[0], pc[(NC - 1) / 2], pc[NC - 1]) /
                    scale_;
            return thrust::make_tuple(points, colors);
        } else {
            return thrust::make_tuple(
                    Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity()),
                    Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity()));
        }
    }
};

struct compute_normals_from_structured_pointcloud_functor {
    compute_normals_from_structured_pointcloud_functor(
            const Eigen::Vector3f *points,
            int width,
            int height)
        : points_(points),
          width_(width),
          height_(height) {};
    const Eigen::Vector3f *points_;
    const int width_;
    const int height_;
    __device__ Eigen::Vector3f operator() (size_t idx) const {
        int i = idx / width_;
        int j = idx % width_;
        if (i < 1 || i >= height_ || j < 1 || j >= width_) {
            return Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        }
        Eigen::Vector3f left = *(points_ + width_ * i + j - 1);
        if (!Eigen::device_all(left.array().isFinite())) {
            left = Eigen::Vector3f::Zero();
        }
        Eigen::Vector3f right = *(points_ + width_ * i + j + 1);
        if (!Eigen::device_all(right.array().isFinite())) {
            right = Eigen::Vector3f::Zero();
        }
        Eigen::Vector3f upper = *(points_ + width_ * (i - 1) + j);
        if (!Eigen::device_all(upper.array().isFinite())) {
            upper = Eigen::Vector3f::Zero();
        }
        Eigen::Vector3f lower = *(points_ + width_ * (i + 1) + j);
        if (!Eigen::device_all(lower.array().isFinite())) {
            lower = Eigen::Vector3f::Zero();
        }
        Eigen::Vector3f hor = left - right;
        Eigen::Vector3f ver = upper - lower;
        Eigen::Vector3f normal = hor.cross(ver);
        float norm = normal.norm();
        if (norm == 0) {
            return Eigen::Vector3f::Zero();
        }
        normal /= norm;
        if (normal.z() > 0) normal *= -1.0f;
        return normal;
    }
};

struct compute_points_from_scan_functor {
    compute_points_from_scan_functor(float min_range, float max_range,
                                     float min_angle, float angle_increment,
                                     int num_steps)
    : min_range_(min_range), max_range_(max_range),
    min_angle_(min_angle), angle_increment_(angle_increment),
    num_steps_(num_steps) {};
    const float min_range_;
    const float max_range_;
    const float min_angle_;
    const float angle_increment_;
    const int num_steps_;
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> operator() (const thrust::tuple<size_t, float, Eigen::Matrix4f_u, float>& x) const {
        size_t idx = thrust::get<0>(x);
        float r = thrust::get<1>(x);
        Eigen::Vector3f color = Eigen::Vector3f::Constant(thrust::get<3>(x));
        if (isnan(r) || r < min_range_ || max_range_ < r) {
            return thrust::make_tuple(Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()),
                                      color);
        }
        Eigen::Matrix4f origin = thrust::get<2>(x);
        int i = idx % num_steps_;
        float angle = min_angle_ + i * angle_increment_;
        Eigen::Vector4f pt = origin * Eigen::Vector4f(r * cos(angle), r * sin(angle), 0.0, 1.0);
        return thrust::make_tuple(pt.head<3>(), color);
    }
};

struct compute_points_from_occvoxels_functor {
    compute_points_from_occvoxels_functor(float voxel_size, int resolution, const Eigen::Vector3f& origin)
    : voxel_size_(voxel_size), resolution_(resolution), origin_(origin) {};
    const float voxel_size_;
    const int resolution_;
    const Eigen::Vector3f origin_;
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> operator() (const OccupancyVoxel& v) const {
        const Eigen::Vector3f pt = (v.grid_index_.cast<float>() + Eigen::Vector3f::Constant(-resolution_ / 2 + 0.5)) * voxel_size_ + origin_;
        return thrust::make_tuple(pt, v.color_);
    }
};

template <typename TC, int NC>
std::shared_ptr<PointCloud> CreatePointCloudFromRGBDImageT(
        const RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic,
        bool project_valid_depth_only,
        float depth_cutoff,
        bool compute_normals) {
    auto pointcloud = std::make_shared<PointCloud>();
    Eigen::Matrix4f camera_pose = extrinsic.inverse();
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    float scale = (sizeof(TC) == 1) ? 255.0 : 1.0;
    int num_valid_pixels = image.depth_.height_ * image.depth_.width_;
    pointcloud->points_.resize(num_valid_pixels);
    pointcloud->colors_.resize(num_valid_pixels);
    convert_from_rgbdimage_functor<TC, NC> func(
            thrust::raw_pointer_cast(image.depth_.data_.data()),
            thrust::raw_pointer_cast(image.color_.data_.data()),
            image.depth_.width_, camera_pose, principal_point, focal_length,
            scale, depth_cutoff);
    thrust::transform(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_valid_pixels),
            make_tuple_begin(pointcloud->points_, pointcloud->colors_), func);
    if (compute_normals) {
        pointcloud->normals_.resize(num_valid_pixels);
        compute_normals_from_structured_pointcloud_functor func_n(
                thrust::raw_pointer_cast(pointcloud->points_.data()),
                image.depth_.width_, image.depth_.height_);
        thrust::transform(thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(num_valid_pixels),
                pointcloud->normals_.begin(), func_n);
    }
    pointcloud->RemoveNoneFinitePoints(project_valid_depth_only, project_valid_depth_only);
    return pointcloud;
}

}  // namespace

std::shared_ptr<PointCloud> PointCloud::CreateFromDepthImage(
        const Image &depth,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic /* = Eigen::Matrix4f::Identity()*/,
        float depth_scale /* = 1000.0*/,
        float depth_trunc /* = 1000.0*/,
        int stride /* = 1*/) {
    if (depth.num_of_channels_ == 1) {
        if (depth.bytes_per_channel_ == 2) {
            auto float_depth =
                    depth.ConvertDepthToFloatImage(depth_scale, depth_trunc);
            return CreatePointCloudFromFloatDepthImage(*float_depth, intrinsic,
                                                       extrinsic, stride);
        } else if (depth.bytes_per_channel_ == 4) {
            return CreatePointCloudFromFloatDepthImage(depth, intrinsic,
                                                       extrinsic, stride);
        }
    }
    utility::LogError(
            "[CreatePointCloudFromDepthImage] Unsupported image format.");
    return std::make_shared<PointCloud>();
}

std::shared_ptr<PointCloud> PointCloud::CreateFromRGBDImage(
        const RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic /* = Eigen::Matrix4f::Identity()*/,
        bool project_valid_depth_only,
        float depth_cutoff,
        bool compute_normals) {
    if (image.color_.bytes_per_channel_ == 1 &&
        image.color_.num_of_channels_ == 3) {
        return CreatePointCloudFromRGBDImageT<uint8_t, 3>(
                image, intrinsic, extrinsic, project_valid_depth_only, depth_cutoff, compute_normals);
    } else if (image.color_.bytes_per_channel_ == 4 &&
               image.color_.num_of_channels_ == 1) {
        return CreatePointCloudFromRGBDImageT<float, 1>(
                image, intrinsic, extrinsic, project_valid_depth_only, depth_cutoff, compute_normals);
    }
    utility::LogError(
            "[CreatePointCloudFromRGBDImage] Unsupported image format.");
    return std::make_shared<PointCloud>();
}

std::shared_ptr<PointCloud> PointCloud::CreateFromLaserScanBuffer(
        const LaserScanBuffer &scan,
        float min_range,
        float max_range) {
    auto pointcloud = std::make_shared<PointCloud>();
    thrust::repeated_range<utility::device_vector<Eigen::Matrix4f_u>::const_iterator>
        range(scan.origins_.begin(), scan.origins_.end(), scan.num_steps_);
    compute_points_from_scan_functor func(min_range, max_range,
                                          scan.min_angle_, scan.GetAngleIncrement(),
                                          scan.num_steps_);
    pointcloud->points_.resize(scan.ranges_.size());
    if (scan.HasIntensities()) {
        pointcloud->colors_.resize(scan.ranges_.size());
        thrust::transform(enumerate_begin(scan.ranges_, range, scan.intensities_),
                          enumerate_end(scan.ranges_, range, scan.intensities_),
                          make_tuple_begin(pointcloud->points_, pointcloud->colors_), func);

    } else {
        thrust::transform(make_tuple_iterator(thrust::make_counting_iterator<size_t>(0),
                                              scan.ranges_.begin(),
                                              range.begin(),
                                              thrust::make_constant_iterator<float>(0)),
                          make_tuple_iterator(thrust::make_counting_iterator(scan.ranges_.size()),
                                              scan.ranges_.end(),
                                              range.end(),
                                              thrust::make_constant_iterator<float>(0)),
                          make_tuple_iterator(pointcloud->points_.begin(), thrust::make_discard_iterator()), func);
    }
    pointcloud->RemoveNoneFinitePoints(true, true);
    return pointcloud;
}

std::shared_ptr<PointCloud> PointCloud::CreateFromOccupancyGrid(
        const OccupancyGrid &occgrid) {
    auto pointcloud = std::make_shared<PointCloud>();
    auto occvoxels = occgrid.ExtractOccupiedVoxels();
    pointcloud->points_.resize(occvoxels->size());
    pointcloud->colors_.resize(occvoxels->size());
    compute_points_from_occvoxels_functor func(occgrid.voxel_size_, occgrid.resolution_, occgrid.origin_);
    thrust::transform(occvoxels->begin(), occvoxels->end(),
                      make_tuple_begin(pointcloud->points_, pointcloud->colors_), func);
    return pointcloud;
}

}
}