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
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/set_operations.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/gather.h>

#include "cupoch/camera/pinhole_camera_parameters.h"
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/geometry_functor.h"
#include "cupoch/geometry/image.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/utility/platform.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct extract_grid_index_functor {
    __device__ Eigen::Vector3i operator()(const Voxel &voxel) const {
        return voxel.grid_index_;
    }
};

__host__ __device__ void GetVoxelBoundingPoints(const Eigen::Vector3f &x,
                                                float r,
                                                Eigen::Vector3f points[8]) {
    points[0] = x + Eigen::Vector3f(-r, -r, -r);
    points[1] = x + Eigen::Vector3f(-r, -r, r);
    points[2] = x + Eigen::Vector3f(r, -r, -r);
    points[3] = x + Eigen::Vector3f(r, -r, r);
    points[4] = x + Eigen::Vector3f(-r, r, -r);
    points[5] = x + Eigen::Vector3f(-r, r, r);
    points[6] = x + Eigen::Vector3f(r, r, -r);
    points[7] = x + Eigen::Vector3f(r, r, r);
}

struct compute_carve_functor {
    compute_carve_functor(const uint8_t *image,
                          int width,
                          int height,
                          int num_of_channels,
                          int bytes_per_channel,
                          float voxel_size,
                          const Eigen::Vector3f &origin,
                          const Eigen::Matrix3f &intrinsic,
                          const Eigen::Matrix3f &rot,
                          const Eigen::Vector3f &trans,
                          bool keep_voxels_outside_image)
        : image_(image),
          width_(width),
          height_(height),
          num_of_channels_(num_of_channels),
          bytes_per_channel_(bytes_per_channel),
          voxel_size_(voxel_size),
          origin_(origin),
          intrinsic_(intrinsic),
          rot_(rot),
          trans_(trans),
          keep_voxels_outside_image_(keep_voxels_outside_image){};
    const uint8_t *image_;
    const int width_;
    const int height_;
    const int num_of_channels_;
    const int bytes_per_channel_;
    const float voxel_size_;
    const Eigen::Vector3f origin_;
    const Eigen::Matrix3f intrinsic_;
    const Eigen::Matrix3f rot_;
    const Eigen::Vector3f trans_;
    bool keep_voxels_outside_image_;
    __device__ bool operator()(
            const thrust::tuple<Eigen::Vector3i, Voxel> &voxel) const {
        bool carve = true;
        float r = voxel_size_ / 2.0;
        const Voxel &vxl = thrust::get<1>(voxel);
        auto x = ((vxl.grid_index_.cast<float>() +
                   Eigen::Vector3f(0.5, 0.5, 0.5)) *
                  voxel_size_) +
                 origin_;
        Eigen::Vector3f pts[8];
        GetVoxelBoundingPoints(x, r, pts);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            auto x_trans = rot_ * pts[i] + trans_;
            auto uvz = intrinsic_ * x_trans;
            float z = uvz(2);
            float u = uvz(0) / z;
            float v = uvz(1) / z;
            float d;
            bool within_boundary;
            thrust::tie(within_boundary, d) =
                    FloatValueAt(image_, u, v, width_, height_,
                                 num_of_channels_, bytes_per_channel_);
            if ((!within_boundary && keep_voxels_outside_image_) ||
                (within_boundary && d > 0 && z >= d)) {
                carve = false;
                break;
            }
        }
        return carve;
    }
};

}  // namespace

VoxelGrid::VoxelGrid() : GeometryBase3D(Geometry::GeometryType::VoxelGrid) {}
VoxelGrid::~VoxelGrid() {}

VoxelGrid::VoxelGrid(const VoxelGrid &src_voxel_grid)
    : GeometryBase3D(Geometry::GeometryType::VoxelGrid),
      voxel_size_(src_voxel_grid.voxel_size_),
      origin_(src_voxel_grid.origin_),
      voxels_keys_(src_voxel_grid.voxels_keys_),
      voxels_values_(src_voxel_grid.voxels_values_) {}

std::pair<thrust::host_vector<Eigen::Vector3i>, thrust::host_vector<Voxel>>
VoxelGrid::GetVoxels() const {
    thrust::host_vector<Eigen::Vector3i> h_keys = voxels_keys_;
    thrust::host_vector<Voxel> h_values = voxels_values_;
    return std::make_pair(h_keys, h_values);
}

void VoxelGrid::SetVoxels(
        const thrust::host_vector<Eigen::Vector3i> &voxels_keys,
        const thrust::host_vector<Voxel> &voxels_values) {
    voxels_keys_ = voxels_keys;
    voxels_values_ = voxels_values;
}

void VoxelGrid::SetVoxels(const std::vector<Eigen::Vector3i> &voxels_keys,
                          const std::vector<Voxel> &voxels_values) {
    voxels_keys_.resize(voxels_keys.size());
    voxels_values_.resize(voxels_values.size());
    copy_host_to_device(voxels_keys, voxels_keys_);
    copy_host_to_device(voxels_values, voxels_values_);
}

VoxelGrid &VoxelGrid::Clear() {
    voxel_size_ = 0.0;
    origin_ = Eigen::Vector3f::Zero();
    voxels_keys_.clear();
    voxels_values_.clear();
    return *this;
}

bool VoxelGrid::IsEmpty() const { return voxels_keys_.empty(); }

Eigen::Vector3f VoxelGrid::GetMinBound() const {
    if (voxels_keys_.empty()) {
        return origin_;
    } else {
        Eigen::Vector3i init = voxels_keys_[0];
        Eigen::Vector3i min_grid_index =
                thrust::reduce(utility::exec_policy(0),
                               voxels_keys_.begin(), voxels_keys_.end(), init,
                               thrust::elementwise_minimum<Eigen::Vector3i>());
        return min_grid_index.cast<float>() * voxel_size_ + origin_;
    }
}

Eigen::Vector3f VoxelGrid::GetMaxBound() const {
    if (voxels_keys_.empty()) {
        return origin_;
    } else {
        Eigen::Vector3i init = voxels_keys_[0];
        Eigen::Vector3i max_grid_index =
                thrust::reduce(utility::exec_policy(0),
                               voxels_keys_.begin(), voxels_keys_.end(), init,
                               thrust::elementwise_maximum<Eigen::Vector3i>());
        return (max_grid_index.cast<float>() + Eigen::Vector3f::Ones()) *
                       voxel_size_ +
               origin_;
    }
}

Eigen::Vector3f VoxelGrid::GetCenter() const {
    Eigen::Vector3f init(0, 0, 0);
    if (voxels_keys_.empty()) {
        return init;
    }
    compute_grid_center_functor func(voxel_size_, origin_);
    Eigen::Vector3f center = thrust::transform_reduce(
            utility::exec_policy(0), voxels_keys_.begin(),
            voxels_keys_.end(), func, init, thrust::plus<Eigen::Vector3f>());
    center /= float(voxels_values_.size());
    return center;
}

AxisAlignedBoundingBox<3> VoxelGrid::GetAxisAlignedBoundingBox() const {
    AxisAlignedBoundingBox<3> box;
    box.min_bound_ = GetMinBound();
    box.max_bound_ = GetMaxBound();
    return box;
}

OrientedBoundingBox VoxelGrid::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
            GetAxisAlignedBoundingBox());
}

VoxelGrid &VoxelGrid::Transform(const Eigen::Matrix4f &transformation) {
    utility::LogError("VoxelGrid::Transform is not supported");
    return *this;
}

VoxelGrid &VoxelGrid::Translate(const Eigen::Vector3f &translation,
                                bool relative) {
    origin_ += translation;
    return *this;
}

VoxelGrid &VoxelGrid::Scale(const float scale, bool center) {
    voxel_size_ *= scale;
    return *this;
}

VoxelGrid &VoxelGrid::Rotate(const Eigen::Matrix3f &R, bool center) {
    utility::LogError("VoxelGrid::Rotate is not supported");
    return *this;
}

VoxelGrid &VoxelGrid::operator+=(const VoxelGrid &voxelgrid) {
    if (voxel_size_ != voxelgrid.voxel_size_) {
        utility::LogError(
                "[VoxelGrid] Could not combine VoxelGrid because voxel_size "
                "differs (this=%f, other=%f)",
                voxel_size_, voxelgrid.voxel_size_);
    }
    if (origin_ != voxelgrid.origin_) {
        utility::LogError(
                "[VoxelGrid] Could not combine VoxelGrid because origin "
                "differs (this=%f,%f,%f, other=%f,%f,%f)",
                origin_(0), origin_(1), origin_(2), voxelgrid.origin_(0),
                voxelgrid.origin_(1), voxelgrid.origin_(2));
    }
    if (this->HasColors() != voxelgrid.HasColors()) {
        utility::LogError(
                "[VoxelGrid] Could not combine VoxelGrid one has colors and "
                "the other not.");
    }
    if (voxelgrid.HasColors()) {
        voxels_keys_.insert(voxels_keys_.end(), voxelgrid.voxels_keys_.begin(),
                            voxelgrid.voxels_keys_.end());
        voxels_values_.insert(voxels_values_.end(),
                              voxelgrid.voxels_values_.begin(),
                              voxelgrid.voxels_values_.end());
        thrust::sort_by_key(utility::exec_policy(0),
                            voxels_keys_.begin(), voxels_keys_.end(),
                            voxels_values_.begin());
        utility::device_vector<int> counts(voxels_keys_.size());
        utility::device_vector<Eigen::Vector3i> new_keys(voxels_keys_.size());
        auto end = thrust::reduce_by_key(
                utility::exec_policy(0), voxels_keys_.begin(),
                voxels_keys_.end(),
                make_tuple_iterator(voxels_values_.begin(),
                                    thrust::make_constant_iterator(1)),
                new_keys.begin(), make_tuple_begin(voxels_values_, counts),
                thrust::equal_to<Eigen::Vector3i>(), add_voxel_color_functor());
        resize_all(thrust::distance(new_keys.begin(), end.first), new_keys,
                   voxels_values_);
        thrust::swap(voxels_keys_, new_keys);
        thrust::transform(voxels_values_.begin(), voxels_values_.end(),
                          counts.begin(), voxels_values_.begin(),
                          divide_voxel_color_functor());
    } else {
        this->AddVoxels(voxelgrid.voxels_values_);
    }
    return *this;
}

VoxelGrid VoxelGrid::operator+(const VoxelGrid &voxelgrid) const {
    return (VoxelGrid(*this) += voxelgrid);
}

void VoxelGrid::AddVoxel(const Voxel &voxel) {
    voxels_keys_.push_back(voxel.grid_index_);
    voxels_values_.push_back(voxel);
    thrust::sort_by_key(utility::exec_policy(0), voxels_keys_.begin(),
                        voxels_keys_.end(), voxels_values_.begin());
    auto end = thrust::unique_by_key(utility::exec_policy(0),
                                     voxels_keys_.begin(), voxels_keys_.end(),
                                     voxels_values_.begin());
    resize_all(thrust::distance(voxels_keys_.begin(), end.first), voxels_keys_,
               voxels_values_);
}

void VoxelGrid::AddVoxels(const utility::device_vector<Voxel> &voxels) {
    voxels_keys_.insert(voxels_keys_.end(),
                        thrust::make_transform_iterator(
                                voxels.begin(), extract_grid_index_functor()),
                        thrust::make_transform_iterator(
                                voxels.end(), extract_grid_index_functor()));
    voxels_values_.insert(voxels_values_.end(), voxels.begin(), voxels.end());
    thrust::sort_by_key(utility::exec_policy(0), voxels_keys_.begin(),
                        voxels_keys_.end(), voxels_values_.begin());
    auto end = thrust::unique_by_key(utility::exec_policy(0),
                                     voxels_keys_.begin(), voxels_keys_.end(),
                                     voxels_values_.begin());
    resize_all(thrust::distance(voxels_keys_.begin(), end.first), voxels_keys_,
               voxels_values_);
}

void VoxelGrid::AddVoxels(const thrust::host_vector<Voxel> &voxels) {
    utility::device_vector<Voxel> voxels_dev = voxels;
    AddVoxels(voxels_dev);
}

VoxelGrid &VoxelGrid::PaintUniformColor(const Eigen::Vector3f &color) {
    thrust::for_each(voxels_values_.begin(), voxels_values_.end(),
                     [c = color] __device__(Voxel & v) { v.color_ = c; });
    return *this;
}

VoxelGrid &VoxelGrid::PaintIndexedColor(
        const utility::device_vector<size_t> &indices,
        const Eigen::Vector3f &color) {
    thrust::for_each(thrust::make_permutation_iterator(voxels_values_.begin(),
                                                       indices.begin()),
                     thrust::make_permutation_iterator(voxels_values_.begin(),
                                                       indices.end()),
                     [c = color] __device__(Voxel & v) { v.color_ = c; });
    return *this;
}

Eigen::Vector3i VoxelGrid::GetVoxel(const Eigen::Vector3f &point) const {
    Eigen::Vector3f voxel_f = (point - origin_) / voxel_size_;
    return (Eigen::floor(voxel_f.array())).cast<int>();
}

Eigen::Vector3f VoxelGrid::GetVoxelCenterCoordinate(
        const Eigen::Vector3i &idx) const {
    auto it = thrust::find(voxels_keys_.begin(), voxels_keys_.end(), idx);
    if (it != voxels_keys_.end()) {
        Eigen::Vector3i voxel_idx = *it;
        return ((voxel_idx.cast<float>() + Eigen::Vector3f(0.5, 0.5, 0.5)) *
                voxel_size_) +
               origin_;
    } else {
        return Eigen::Vector3f::Zero();
    }
}

std::array<Eigen::Vector3f, 8> VoxelGrid::GetVoxelBoundingPoints(
        const Eigen::Vector3i &index) const {
    float r = voxel_size_ / 2.0;
    auto x = GetVoxelCenterCoordinate(index);
    std::array<Eigen::Vector3f, 8> points;
    ::GetVoxelBoundingPoints(x, r, points.data());
    return points;
}

std::vector<bool> VoxelGrid::CheckIfIncluded(
        const std::vector<Eigen::Vector3f> &queries) {
    std::vector<bool> output;
    output.resize(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        auto query = GetVoxel(queries[i]);
        auto itr =
                thrust::find(voxels_keys_.begin(), voxels_keys_.end(), query);
        output[i] = (itr != voxels_keys_.end());
    }
    return output;
}

VoxelGrid &VoxelGrid::CarveDepthMap(
        const Image &depth_map,
        const camera::PinholeCameraParameters &camera_parameter,
        bool keep_voxels_outside_image) {
    if (depth_map.height_ != camera_parameter.intrinsic_.height_ ||
        depth_map.width_ != camera_parameter.intrinsic_.width_) {
        utility::LogError(
                "[VoxelGrid] provided depth_map dimensions are not compatible "
                "with the provided camera_parameters");
    }

    auto rot = camera_parameter.extrinsic_.block<3, 3>(0, 0);
    auto trans = camera_parameter.extrinsic_.block<3, 1>(0, 3);
    auto intrinsic = camera_parameter.intrinsic_.intrinsic_matrix_;

    // get for each voxel if it projects to a valid pixel and check if the voxel
    // depth is behind the depth of the depth map at the projected pixel.
    compute_carve_functor func(
            thrust::raw_pointer_cast(depth_map.data_.data()), depth_map.width_,
            depth_map.height_, depth_map.num_of_channels_,
            depth_map.bytes_per_channel_, voxel_size_, origin_, intrinsic, rot,
            trans, keep_voxels_outside_image);
    remove_if_vectors(utility::exec_policy(0), func, voxels_keys_,
                      voxels_values_);
    return *this;
}

VoxelGrid &VoxelGrid::CarveSilhouette(
        const Image &silhouette_mask,
        const camera::PinholeCameraParameters &camera_parameter,
        bool keep_voxels_outside_image) {
    if (silhouette_mask.height_ != camera_parameter.intrinsic_.height_ ||
        silhouette_mask.width_ != camera_parameter.intrinsic_.width_) {
        utility::LogError(
                "[VoxelGrid] provided silhouette_mask dimensions are not "
                "compatible with the provided camera_parameters");
    }

    auto rot = camera_parameter.extrinsic_.block<3, 3>(0, 0);
    auto trans = camera_parameter.extrinsic_.block<3, 1>(0, 3);
    auto intrinsic = camera_parameter.intrinsic_.intrinsic_matrix_;

    // get for each voxel if it projects to a valid pixel and check if the pixel
    // is set (>0).
    compute_carve_functor func(
            thrust::raw_pointer_cast(silhouette_mask.data_.data()),
            silhouette_mask.width_, silhouette_mask.height_,
            silhouette_mask.num_of_channels_,
            silhouette_mask.bytes_per_channel_, voxel_size_, origin_, intrinsic,
            rot, trans, keep_voxels_outside_image);
    remove_if_vectors(func, voxels_keys_, voxels_values_);
    return *this;
}

std::shared_ptr<VoxelGrid> VoxelGrid::SelectByIndex(
                       const utility::device_vector<size_t> &indices, 
                       bool invert) {
    auto dst = std::make_shared<VoxelGrid>();
    if (invert) {
        size_t n_out = voxels_values_.size() - indices.size();
        utility::device_vector<size_t> sorted_indices = indices;
        thrust::sort(utility::exec_policy(0), sorted_indices.begin(),
                        sorted_indices.end());
        utility::device_vector<size_t> inv_indices(n_out);
        thrust::set_difference(thrust::make_counting_iterator<size_t>(0),
                                thrust::make_counting_iterator(voxels_values_.size()),
                                sorted_indices.begin(), sorted_indices.end(),
                                inv_indices.begin());

        dst->voxels_values_.resize(inv_indices.size());
        dst->voxels_keys_.resize(inv_indices.size());
        dst->voxel_size_ = voxel_size_;
        dst->origin_ = origin_;

        thrust::gather(utility::exec_policy(utility::GetStream(0)),
                    inv_indices.begin(), inv_indices.end(), voxels_values_.begin(),
                    dst->voxels_values_.begin());
        thrust::gather(utility::exec_policy(utility::GetStream(0)),
                    inv_indices.begin(), inv_indices.end(), voxels_keys_.begin(),
                    dst->voxels_keys_.begin());
        cudaSafeCall(cudaDeviceSynchronize());

    } else {
        dst->voxels_values_.resize(indices.size());
        dst->voxels_keys_.resize(indices.size());
        dst->voxel_size_ = voxel_size_;
        dst->origin_ = origin_;

        thrust::gather(utility::exec_policy(utility::GetStream(0)),
                    indices.begin(), indices.end(), voxels_values_.begin(),
                    dst->voxels_values_.begin());
        thrust::gather(utility::exec_policy(utility::GetStream(0)),
                    indices.begin(), indices.end(), voxels_keys_.begin(),
                    dst->voxels_keys_.begin());
        cudaSafeCall(cudaDeviceSynchronize());
    }
    return dst;
}
