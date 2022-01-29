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
#include <thrust/iterator/discard_iterator.h>

#include "cupoch/integration/integrate_functor.h"
#include "cupoch/integration/marching_cubes_const.h"
#include "cupoch/integration/uniform_tsdfvolume.h"
#include "cupoch/utility/eigen.h"
#include "cupoch/utility/helper.h"

namespace cupoch {
namespace integration {

namespace {

__device__ float GetTSDFAt(const Eigen::Vector3f &p,
                           const geometry::TSDFVoxel *voxels,
                           float voxel_length,
                           int resolution) {
    Eigen::Vector3i idx;
    Eigen::Vector3f p_grid = p / voxel_length - Eigen::Vector3f(0.5, 0.5, 0.5);
    for (int i = 0; i < 3; i++) {
        idx(i) = (int)floorf(p_grid(i));
    }
    Eigen::Vector3f r = p_grid - idx.cast<float>();

    float tsdf = 0;
    tsdf += (1 - r(0)) * (1 - r(1)) * (1 - r(2)) *
            voxels[IndexOf(idx + Eigen::Vector3i(0, 0, 0), resolution)].tsdf_;
    tsdf += (1 - r(0)) * (1 - r(1)) * r(2) *
            voxels[IndexOf(idx + Eigen::Vector3i(0, 0, 1), resolution)].tsdf_;
    tsdf += (1 - r(0)) * r(1) * (1 - r(2)) *
            voxels[IndexOf(idx + Eigen::Vector3i(0, 1, 0), resolution)].tsdf_;
    tsdf += (1 - r(0)) * r(1) * r(2) *
            voxels[IndexOf(idx + Eigen::Vector3i(0, 1, 1), resolution)].tsdf_;
    tsdf += r(0) * (1 - r(1)) * (1 - r(2)) *
            voxels[IndexOf(idx + Eigen::Vector3i(1, 0, 0), resolution)].tsdf_;
    tsdf += r(0) * (1 - r(1)) * r(2) *
            voxels[IndexOf(idx + Eigen::Vector3i(1, 0, 1), resolution)].tsdf_;
    tsdf += r(0) * r(1) * (1 - r(2)) *
            voxels[IndexOf(idx + Eigen::Vector3i(1, 1, 0), resolution)].tsdf_;
    tsdf += r(0) * r(1) * r(2) *
            voxels[IndexOf(idx + Eigen::Vector3i(1, 1, 1), resolution)].tsdf_;
    return tsdf;
}

__device__ Eigen::Vector3f GetNormalAt(const Eigen::Vector3f &p,
                                       const geometry::TSDFVoxel *voxels,
                                       float voxel_length,
                                       int resolution) {
    Eigen::Vector3f n;
    const double half_gap = 0.99 * voxel_length;
#pragma unroll
    for (int i = 0; i < 3; i++) {
        Eigen::Vector3f p0 = p;
        p0(i) -= half_gap;
        Eigen::Vector3f p1 = p;
        p1(i) += half_gap;
        n(i) = GetTSDFAt(p1, voxels, voxel_length, resolution) -
               GetTSDFAt(p0, voxels, voxel_length, resolution);
    }
    return n.normalized();
}

struct extract_pointcloud_functor {
    extract_pointcloud_functor(const geometry::TSDFVoxel *voxels,
                               int resolution,
                               float voxel_length,
                               const Eigen::Vector3f &origin,
                               TSDFVolumeColorType color_type)
        : voxels_(voxels),
          resolution_(resolution),
          voxel_length_(voxel_length),
          origin_(origin),
          half_voxel_length_(0.5 * voxel_length_),
          color_type_(color_type){};
    const geometry::TSDFVoxel *voxels_;
    const int resolution_;
    const float voxel_length_;
    const Eigen::Vector3f origin_;
    const float half_voxel_length_;
    const TSDFVolumeColorType color_type_;
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>
    operator()(const size_t idx) {
        int res2 = (resolution_ - 2) * (resolution_ - 2);
        int x = idx / (3 * res2) + 1;
        int yzi = idx % (3 * res2);
        int y = yzi / (3 * (resolution_ - 2)) + 1;
        int zi = yzi % (3 * (resolution_ - 2));
        int z = zi / 3 + 1;
        int i = zi % 3;

        Eigen::Vector3f point = Eigen::Vector3f::Constant(
                std::numeric_limits<float>::quiet_NaN());
        Eigen::Vector3f normal = Eigen::Vector3f::Constant(
                std::numeric_limits<float>::quiet_NaN());
        Eigen::Vector3f color = Eigen::Vector3f::Constant(
                std::numeric_limits<float>::quiet_NaN());
        Eigen::Vector3i idx0(x, y, z);
        Eigen::Vector3f h_res =
                Eigen::Vector3f::Constant(resolution_ / 2) * voxel_length_;
        float w0 = voxels_[IndexOf(idx0, resolution_)].weight_;
        float f0 = voxels_[IndexOf(idx0, resolution_)].tsdf_;
        const Eigen::Vector3f &c0 = voxels_[IndexOf(idx0, resolution_)].color_;
        if (!(w0 != 0.0f && f0 < 0.98f && f0 >= -0.98f)) {
            return thrust::make_tuple(point, normal, color);
        }
        Eigen::Vector3f p0(half_voxel_length_ + voxel_length_ * x,
                           half_voxel_length_ + voxel_length_ * y,
                           half_voxel_length_ + voxel_length_ * z);
        Eigen::Vector3f p1 = p0;
        p1(i) += voxel_length_;
        Eigen::Vector3i idx1 = idx0;
        idx1(i) += 1;
        if (idx1(i) < resolution_ - 1) {
            float w1 = voxels_[IndexOf(idx1, resolution_)].weight_;
            float f1 = voxels_[IndexOf(idx1, resolution_)].tsdf_;
            const Eigen::Vector3f &c1 =
                    voxels_[IndexOf(idx1, resolution_)].color_;
            if (w1 != 0.0f && f1 < 0.98f && f1 >= -0.98f && f0 * f1 < 0) {
                float r0 = std::fabs(f0);
                float r1 = std::fabs(f1);
                Eigen::Vector3f p = p0;
                p(i) = (p0(i) * r1 + p1(i) * r0) / (r0 + r1);
                point = p + origin_;
                if (color_type_ == TSDFVolumeColorType::RGB8) {
                    color = (c0 * r1 + c1 * r0) / (r0 + r1) / 255.0f;
                } else if (color_type_ == TSDFVolumeColorType::Gray32) {
                    color = (c0 * r1 + c1 * r0) / (r0 + r1);
                }
                // has_normal
                normal = GetNormalAt(p, voxels_, voxel_length_, resolution_);
            }
        }
        return thrust::make_tuple(point - h_res, normal, color);
    }
};

struct count_valid_voxels_functor {
    count_valid_voxels_functor(const geometry::TSDFVoxel *voxels,
                               int resolution)
        : voxels_(voxels), resolution_(resolution){};
    const geometry::TSDFVoxel *voxels_;
    const int resolution_;
    __device__ bool operator()(
            const thrust::tuple<size_t, geometry::TSDFVoxel> &kv) const {
        size_t idx = thrust::get<0>(kv);
        int x, y, z;
        thrust::tie(x, y, z) = KeyOf(idx, resolution_);
        if (x == resolution_ - 1 || y == resolution_ - 1 ||
            z == resolution_ - 1)
            return false;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            Eigen::Vector3i idx = Eigen::Vector3i(
                    x + shift[i][0], y + shift[i][1], z + shift[i][2]);
            if (voxels_[IndexOf(idx, resolution_)].weight_ == 0.0f)
                return false;
        }
        return true;
    }
};

struct extract_mesh_phase0_functor {
    extract_mesh_phase0_functor(const geometry::TSDFVoxel *voxels,
                                int resolution)
        : voxels_(voxels), resolution_(resolution){};
    const geometry::TSDFVoxel *voxels_;
    const int resolution_;
    __device__ thrust::tuple<Eigen::Vector3i, int> operator()(size_t idx) {
        int x, y, z;
        thrust::tie(x, y, z) = KeyOf(idx, resolution_ - 1);

        int cube_index = 0;
        Eigen::Vector3i key = Eigen::Vector3i(x, y, z);
        for (int i = 0; i < 8; ++i) {
            Eigen::Vector3i idxs =
                    key +
                    Eigen::Vector3i(shift[i][0], shift[i][1], shift[i][2]);
            if (voxels_[IndexOf(idxs, resolution_)].weight_ == 0.0f) {
                return thrust::make_tuple(key, -1);
            } else {
                float f = voxels_[IndexOf(idxs, resolution_)].tsdf_;
                if (f < 0.0f) {
                    cube_index |= (1 << i);
                }
            }
        }
        return thrust::make_tuple(key, cube_index);
    }
};

struct extract_mesh_phase1_functor {
    extract_mesh_phase1_functor(const geometry::TSDFVoxel *voxels,
                                const Eigen::Vector3i *keys,
                                int resolution,
                                TSDFVolumeColorType color_type)
        : voxels_(voxels),
          keys_(keys),
          resolution_(resolution),
          color_type_(color_type){};
    const geometry::TSDFVoxel *voxels_;
    const Eigen::Vector3i *keys_;
    const int resolution_;
    TSDFVolumeColorType color_type_;
    __device__ thrust::tuple<float, Eigen::Vector3f> operator()(size_t idx) {
        int j = idx / 8;
        int i = idx % 8;
        const Eigen::Vector3i &key = keys_[j];
        Eigen::Vector3i idxs =
                key + Eigen::Vector3i(shift[i][0], shift[i][1], shift[i][2]);
        const geometry::TSDFVoxel v = voxels_[IndexOf(idxs, resolution_)];
        Eigen::Vector3f c = Eigen::Vector3f::Zero();
        if (v.weight_ == 0.0f) {
            return thrust::make_tuple(0.0f, c);
        } else {
            float f = v.tsdf_;
            if (color_type_ == TSDFVolumeColorType::RGB8) {
                c = v.color_ / 255.0;
            } else if (color_type_ == TSDFVolumeColorType::Gray32) {
                c = v.color_;
            }
            return thrust::make_tuple(f, c);
        }
    }
};

struct extract_mesh_phase2_functor {
    extract_mesh_phase2_functor(const Eigen::Vector3i *keys,
                                const int *cube_indices,
                                const Eigen::Vector3f &origin,
                                int resolution,
                                float voxel_length,
                                const float *fs,
                                const Eigen::Vector3f *cs,
                                TSDFVolumeColorType color_type)
        : keys_(keys),
          cube_indices_(cube_indices),
          origin_(origin),
          resolution_(resolution),
          voxel_length_(voxel_length),
          half_voxel_length_(0.5 * voxel_length_),
          fs_(fs),
          cs_(cs),
          color_type_(color_type){};
    const Eigen::Vector3i *keys_;
    const int *cube_indices_;
    const Eigen::Vector3f origin_;
    const int resolution_;
    const float voxel_length_;
    const float half_voxel_length_;
    const float *fs_;
    const Eigen::Vector3f *cs_;
    const TSDFVolumeColorType color_type_;
    __device__ thrust::
            tuple<Eigen::Vector3i, int, int, Eigen::Vector3f, Eigen::Vector3f>
            operator()(size_t idx) const {
        int j = idx / 12;
        const Eigen::Vector3i &xyz = keys_[j];
        int cube_index = cube_indices_[j];
        int offset = j * 8;
        int x = xyz[0];
        int y = xyz[1];
        int z = xyz[2];
        int i = idx % 12;
        if (edge_table[cube_index] & (1 << i)) {
            Eigen::Vector4i edge_index =
                    Eigen::Vector4i(x, y, z, 0) +
                    Eigen::Vector4i(edge_shift[i][0], edge_shift[i][1],
                                    edge_shift[i][2], edge_shift[i][3]);
            Eigen::Vector3f pt(
                    half_voxel_length_ + voxel_length_ * edge_index(0),
                    half_voxel_length_ + voxel_length_ * edge_index(1),
                    half_voxel_length_ + voxel_length_ * edge_index(2));
            float f0 = abs(fs_[offset + edge_to_vert[i][0]]);
            float f1 = abs(fs_[offset + edge_to_vert[i][1]]);
            pt(edge_index(3)) += f0 * voxel_length_ / (f0 + f1);
            Eigen::Vector3f vertex = pt + origin_;
            Eigen::Vector3f vertex_color = Eigen::Vector3f::Zero();
            if (color_type_ != TSDFVolumeColorType::NoColor) {
                const auto &c0 = cs_[offset + edge_to_vert[i][0]];
                const auto &c1 = cs_[offset + edge_to_vert[i][1]];
                vertex_color = (f1 * c0 + f0 * c1) / (f0 + f1);
            }
            return thrust::make_tuple(
                    xyz, cube_index, i,
                    vertex - Eigen::Vector3f::Constant(resolution_ / 2) *
                                     voxel_length_,
                    vertex_color);
        } else {
            Eigen::Vector3i index = -Eigen::Vector3i::Ones();
            Eigen::Vector3f vertex = Eigen::Vector3f::Zero();
            Eigen::Vector3f vertex_color = Eigen::Vector3f::Zero();
            return thrust::make_tuple(index, cube_index, i, vertex,
                                      vertex_color);
        }
    }
};

__constant__ int vert_table[3] = {0, 2, 1};

struct extract_mesh_phase3_functor {
    extract_mesh_phase3_functor(const int *cube_index,
                                const int *vert_no,
                                const int *key_index,
                                Eigen::Vector3i *triangles)
        : cube_index_(cube_index),
          vert_no_(vert_no),
          key_index_(key_index),
          triangles_(triangles){};
    const int *cube_index_;
    const int *vert_no_;
    const int *key_index_;
    Eigen::Vector3i *triangles_;
    __device__ void operator()(size_t idx) {
        const int kindx0 = key_index_[idx];
        const int kindx1 = key_index_[idx + 1];
        for (int j = kindx0; j < kindx1; ++j) {
            const int cindx = cube_index_[j];
            for (int i = 0; tri_table[cindx][i] != -1; ++i) {
                const int tri_idx = tri_table[cindx][i];
                for (int l = kindx0; l < kindx1; ++l) {
                    if (vert_no_[l] == tri_idx) {
                        triangles_[idx * 4 + i / 3][vert_table[i % 3]] = l;
                    }
                }
            }
        }
    }
};

struct extract_voxel_pointcloud_functor {
    extract_voxel_pointcloud_functor(const Eigen::Vector3f &origin,
                                     int resolution,
                                     float voxel_length)
        : origin_(origin),
          resolution_(resolution),
          voxel_length_(voxel_length),
          half_voxel_length_(0.5 * voxel_length){};
    const Eigen::Vector3f origin_;
    const int resolution_;
    const float voxel_length_;
    const float half_voxel_length_;
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> operator()(
            const thrust::tuple<size_t, geometry::TSDFVoxel> &kv) {
        int idx = thrust::get<0>(kv);
        int x, y, z;
        int h_res = resolution_ / 2;
        thrust::tie(x, y, z) = KeyOf(idx, resolution_);
        geometry::TSDFVoxel v = thrust::get<1>(kv);
        Eigen::Vector3f pt(half_voxel_length_ + voxel_length_ * (x - h_res),
                           half_voxel_length_ + voxel_length_ * (y - h_res),
                           half_voxel_length_ + voxel_length_ * (z - h_res));
        if (v.weight_ != 0.0f && v.tsdf_ < 0.98f && v.tsdf_ >= -0.98f) {
            float c = (v.tsdf_ + 1.0) * 0.5;
            return thrust::make_tuple(pt + origin_, Eigen::Vector3f(c, c, c));
        }
        return thrust::make_tuple(
                Eigen::Vector3f::Constant(
                        std::numeric_limits<float>::quiet_NaN()),
                Eigen::Vector3f::Constant(
                        std::numeric_limits<float>::quiet_NaN()));
    }
};

struct extract_voxel_grid_functor {
    extract_voxel_grid_functor(int resolution) : resolution_(resolution){};
    const int resolution_;
    __device__ thrust::tuple<Eigen::Vector3i, geometry::Voxel> operator()(
            const thrust::tuple<size_t, geometry::TSDFVoxel> &kv) {
        int idx = thrust::get<0>(kv);
        int x, y, z;
        thrust::tie(x, y, z) = KeyOf(idx, resolution_);
        Eigen::Vector3i grid_idx = Eigen::Vector3i(x, y, z);
        geometry::TSDFVoxel v = thrust::get<1>(kv);
        const float w = v.weight_;
        const float f = v.tsdf_;
        if (w != 0.0f && f < 0.98f && f >= -0.98f) {
            float c = (f + 1.0) * 0.5;
            return thrust::make_tuple(
                    grid_idx,
                    geometry::Voxel(grid_idx, Eigen::Vector3f(c, c, c)));
        }
        return thrust::make_tuple(
                Eigen::Vector3i::Constant(geometry::INVALID_VOXEL_INDEX),
                geometry::Voxel());
    }
};

struct raycast_tsdf_functor {
    raycast_tsdf_functor(const geometry::TSDFVoxel *voxels,
                         int width,
                         float fx,
                         float fy,
                         float cx,
                         float cy,
                         const Eigen::Matrix4f &campose,
                         float voxel_length,
                         int resolution,
                         const Eigen::Vector3f &origin,
                         float sdf_trunc,
                         TSDFVolumeColorType color_type)
        : voxels_(voxels),
          width_(width),
          fx_(fx),
          fy_(fy),
          cx_(cx),
          cy_(cy),
          campose_(campose),
          voxel_length_(voxel_length),
          resolution_(resolution),
          origin_(origin),
          sdf_trunc_(sdf_trunc),
          color_type_(color_type){};
    const geometry::TSDFVoxel *voxels_;
    const int width_;
    const float fx_;
    const float fy_;
    const float cx_;
    const float cy_;
    const Eigen::Matrix4f campose_;
    const float voxel_length_;
    const int resolution_;
    const Eigen::Vector3f origin_;
    const float sdf_trunc_;
    const TSDFVolumeColorType color_type_;
    __device__ __forceinline__ float InterpolateTrilinearly(
            const Eigen::Vector3f &point,
            const geometry::TSDFVoxel *voxels,
            int resolution) const {
        Eigen::Vector3i point_in_grid = point.cast<int>();
        const float vx = (float)point_in_grid[0] + 0.5f;
        const float vy = (float)point_in_grid[1] + 0.5f;
        const float vz = (float)point_in_grid[2] + 0.5f;
        point_in_grid[0] =
                (point[0] < vx) ? (point_in_grid[0] - 1) : point_in_grid[0];
        point_in_grid[1] =
                (point[1] < vy) ? (point_in_grid[1] - 1) : point_in_grid[1];
        point_in_grid[2] =
                (point[2] < vz) ? (point_in_grid[2] - 1) : point_in_grid[2];
        const float a = point[0] - ((float)point_in_grid[0] + 0.5f);
        const float b = point[1] - ((float)point_in_grid[1] + 0.5f);
        const float c = point[2] - ((float)point_in_grid[2] + 0.5f);
        geometry::TSDFVoxel v0 = voxels[IndexOf(point_in_grid, resolution)];
        geometry::TSDFVoxel v1 = voxels[IndexOf(
                point_in_grid + Eigen::Vector3i(0, 0, 1), resolution)];
        geometry::TSDFVoxel v2 = voxels[IndexOf(
                point_in_grid + Eigen::Vector3i(0, 1, 0), resolution)];
        geometry::TSDFVoxel v3 = voxels[IndexOf(
                point_in_grid + Eigen::Vector3i(0, 1, 1), resolution)];
        geometry::TSDFVoxel v4 = voxels[IndexOf(
                point_in_grid + Eigen::Vector3i(1, 0, 0), resolution)];
        geometry::TSDFVoxel v5 = voxels[IndexOf(
                point_in_grid + Eigen::Vector3i(1, 0, 1), resolution)];
        geometry::TSDFVoxel v6 = voxels[IndexOf(
                point_in_grid + Eigen::Vector3i(1, 1, 0), resolution)];
        geometry::TSDFVoxel v7 = voxels[IndexOf(
                point_in_grid + Eigen::Vector3i(1, 1, 1), resolution)];
        return v0.tsdf_ * (1 - a) * (1 - b) * (1 - c) +
               v1.tsdf_ * (1 - a) * (1 - b) * c +
               v2.tsdf_ * (1 - a) * b * (1 - c) + v3.tsdf_ * (1 - a) * b * c +
               v4.tsdf_ * a * (1 - b) * (1 - c) + v5.tsdf_ * a * (1 - b) * c +
               v6.tsdf_ * a * b * (1 - c) + v7.tsdf_ * a * b * c;
    }
    __device__ __forceinline__ float GetMinTime(
            float length,
            const Eigen::Vector3f &origin,
            const Eigen::Vector3f &direction) const {
        float txmin =
                ((direction[0] > 0 ? 0.0f : length) - origin[0]) / direction[0];
        float tymin =
                ((direction[1] > 0 ? 0.0f : length) - origin[1]) / direction[1];
        float tzmin =
                ((direction[2] > 0 ? 0.0f : length) - origin[2]) / direction[2];
        return fmax(fmax(txmin, tymin), tzmin);
    }
    __device__ __forceinline__ float GetMaxTime(
            float length,
            const Eigen::Vector3f &origin,
            const Eigen::Vector3f &direction) const {
        float txmax =
                ((direction[0] > 0 ? length : 0.0f) - origin[0]) / direction[0];
        float tymax =
                ((direction[1] > 0 ? length : 0.0f) - origin[1]) / direction[1];
        float tzmax =
                ((direction[2] > 0 ? length : 0.0f) - origin[2]) / direction[2];
        return fmin(fmin(txmax, tymax), tzmax);
    }
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>
    operator()(size_t idx) const {
        const int y = idx / width_;
        const int x = idx % width_;
        const float length = resolution_ * voxel_length_;
        const Eigen::Vector3f pixel_pos((x - cx_) / fx_, (y - cy_) / fy_, 1.0f);
        Eigen::Vector3f ray_dir = campose_.block<3, 3>(0, 0) * pixel_pos;
        Eigen::Vector3i h_res = Eigen::Vector3i::Constant(resolution_ / 2);
        float ray_dir_norm = ray_dir.norm();
        if (ray_dir_norm == 0) {
            return thrust::make_tuple(
                    Eigen::Vector3f::Constant(
                            std::numeric_limits<float>::quiet_NaN()),
                    Eigen::Vector3f::Constant(
                            std::numeric_limits<float>::quiet_NaN()),
                    Eigen::Vector3f::Constant(
                            std::numeric_limits<float>::quiet_NaN()));
        }
        ray_dir /= ray_dir_norm;
        Eigen::Vector3f t = campose_.block<3, 1>(0, 3) - origin_;
        float ray_len = fmax(GetMinTime(length, t, ray_dir), 0.0f);
        if (ray_len >= GetMaxTime(length, t, ray_dir)) {
            return thrust::make_tuple(
                    Eigen::Vector3f::Constant(
                            std::numeric_limits<float>::quiet_NaN()),
                    Eigen::Vector3f::Constant(
                            std::numeric_limits<float>::quiet_NaN()),
                    Eigen::Vector3f::Constant(
                            std::numeric_limits<float>::quiet_NaN()));
        }
        ray_len += voxel_length_;
        Eigen::Vector3i grid_idx =
                Eigen::device_vectorize<float, 3, ::floor>(
                        (t + (ray_dir * ray_len)) / voxel_length_)
                        .cast<int>() +
                h_res;
        if (grid_idx[0] < 0 || grid_idx[0] >= resolution_ - 1 ||
            grid_idx[1] < 0 || grid_idx[1] >= resolution_ - 1 ||
            grid_idx[2] < 0 || grid_idx[2] >= resolution_ - 1) {
            return thrust::make_tuple(
                    Eigen::Vector3f::Constant(
                            std::numeric_limits<float>::quiet_NaN()),
                    Eigen::Vector3f::Constant(
                            std::numeric_limits<float>::quiet_NaN()),
                    Eigen::Vector3f::Constant(
                            std::numeric_limits<float>::quiet_NaN()));
        }
        geometry::TSDFVoxel v = voxels_[IndexOf(grid_idx, resolution_)];
        const float max_search_length = ray_len + length * sqrt(2.0f);
        for (; ray_len < max_search_length; ray_len += sdf_trunc_ * 0.5f) {
            grid_idx = Eigen::device_vectorize<float, 3, ::floor>(
                               (t + (ray_dir * (ray_len + sdf_trunc_ * 0.5f))) /
                               voxel_length_)
                               .cast<int>() +
                       h_res;
            if (grid_idx[0] < 1 || grid_idx[0] >= resolution_ - 1 ||
                grid_idx[1] < 1 || grid_idx[1] >= resolution_ - 1 ||
                grid_idx[2] < 1 || grid_idx[2] >= resolution_ - 1)
                continue;
            const geometry::TSDFVoxel prev_v = v;
            v = voxels_[IndexOf(grid_idx, resolution_)];
            if (prev_v.tsdf_ < 0.0f && v.tsdf_ > 0.0f) break;
            if (prev_v.tsdf_ > 0.0f && v.tsdf_ < 0.0f) {
                const float t_star = ray_len - sdf_trunc_ * 0.5f *
                                                       prev_v.tsdf_ /
                                                       (v.tsdf_ - prev_v.tsdf_);
                const Eigen::Vector3f vertex = t + ray_dir * t_star;
                const Eigen::Vector3f loc_in_grid =
                        vertex / voxel_length_ + h_res.cast<float>();
                if (loc_in_grid[0] < 1 || loc_in_grid[0] >= resolution_ - 1 ||
                    loc_in_grid[1] < 1 || loc_in_grid[1] >= resolution_ - 1 ||
                    loc_in_grid[2] < 1 || loc_in_grid[2] >= resolution_ - 1)
                    break;
                Eigen::Vector3f normal;
                Eigen::Vector3f shifted = loc_in_grid;
                shifted[0] += 1;
                if (shifted[0] >= resolution_ - 1) break;
                const float fx1 =
                        InterpolateTrilinearly(shifted, voxels_, resolution_);
                shifted = loc_in_grid;
                shifted[0] -= 1;
                if (shifted[0] < 1) break;
                const float fx2 =
                        InterpolateTrilinearly(shifted, voxels_, resolution_);
                normal[0] = fx1 - fx2;

                shifted = loc_in_grid;
                shifted[1] += 1;
                if (shifted[1] >= resolution_ - 1) break;
                const float fy1 =
                        InterpolateTrilinearly(shifted, voxels_, resolution_);
                shifted = loc_in_grid;
                shifted[1] -= 1;
                if (shifted[1] < 1) break;
                const float fy2 =
                        InterpolateTrilinearly(shifted, voxels_, resolution_);
                normal[1] = fy1 - fy2;

                shifted = loc_in_grid;
                shifted[2] += 1;
                if (shifted[2] >= resolution_ - 1) break;
                const float fz1 =
                        InterpolateTrilinearly(shifted, voxels_, resolution_);
                shifted = loc_in_grid;
                shifted[2] -= 1;
                if (shifted[2] < 1) break;
                const float fz2 =
                        InterpolateTrilinearly(shifted, voxels_, resolution_);
                normal[2] = fz1 - fz2;
                const float norm_nml = normal.norm();
                if (norm_nml == 0) break;
                normal /= norm_nml;
                const geometry::TSDFVoxel v =
                        voxels_[IndexOf(loc_in_grid.cast<int>(), resolution_)];
                Eigen::Vector3f c = Eigen::Vector3f::Zero();
                if (color_type_ == TSDFVolumeColorType::RGB8) {
                    c = v.color_ / 255.0;
                } else if (color_type_ == TSDFVolumeColorType::Gray32) {
                    c = v.color_;
                }
                return thrust::make_tuple(vertex + origin_, normal, c);
            }
        }
        return thrust::make_tuple(
                Eigen::Vector3f::Constant(
                        std::numeric_limits<float>::quiet_NaN()),
                Eigen::Vector3f::Constant(
                        std::numeric_limits<float>::quiet_NaN()),
                Eigen::Vector3f::Constant(
                        std::numeric_limits<float>::quiet_NaN()));
    }
};

}  // namespace

UniformTSDFVolume::UniformTSDFVolume(
        float length,
        int resolution,
        float sdf_trunc,
        TSDFVolumeColorType color_type,
        const Eigen::Vector3f &origin /* = Eigen::Vector3f::Zero()*/)
    : TSDFVolume(length / (float)resolution, sdf_trunc, color_type),
      origin_(origin),
      length_(length),
      resolution_(resolution),
      voxel_num_(resolution * resolution * resolution) {
    voxels_.resize(voxel_num_);
}

UniformTSDFVolume::~UniformTSDFVolume() {}

UniformTSDFVolume::UniformTSDFVolume(const UniformTSDFVolume &other)
    : TSDFVolume(other),
      voxels_(other.voxels_),
      origin_(other.origin_),
      length_(other.length_),
      resolution_(other.resolution_),
      voxel_num_(other.voxel_num_) {}

void UniformTSDFVolume::Reset() { voxels_.clear(); }

void UniformTSDFVolume::Integrate(
        const geometry::RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic) {
    // This function goes through the voxels, and scan convert the relative
    // depth/color value into the voxel.
    // The following implementation is a highly optimized version.
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
                "[UniformTSDFVolume::Integrate] Unsupported image format.");
    }
    auto depth2cameradistance =
            geometry::Image::CreateDepthToCameraDistanceMultiplierFloatImage(
                    intrinsic);
    IntegrateWithDepthToCameraDistanceMultiplier(image, intrinsic, extrinsic,
                                                 *depth2cameradistance);
}

std::shared_ptr<geometry::PointCloud> UniformTSDFVolume::ExtractPointCloud() {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    size_t n_valid_voxels =
            thrust::count_if(voxels_.begin(), voxels_.end(),
                             [] __device__(const geometry::TSDFVoxel &v) {
                                 return (v.weight_ != 0.0f && v.tsdf_ < 0.98f &&
                                         v.tsdf_ >= -0.98f);
                             });
    extract_pointcloud_functor func(thrust::raw_pointer_cast(voxels_.data()),
                                    resolution_, voxel_length_, origin_,
                                    color_type_);
    pointcloud->points_.resize(n_valid_voxels);
    pointcloud->normals_.resize(n_valid_voxels);
    pointcloud->colors_.resize(n_valid_voxels);
    size_t n_total =
            (resolution_ - 2) * (resolution_ - 2) * (resolution_ - 2) * 3;
    auto begin = make_tuple_begin(pointcloud->points_, pointcloud->normals_,
                                  pointcloud->colors_);
    auto end_p = thrust::copy_if(
            thrust::make_transform_iterator(
                    thrust::make_counting_iterator<size_t>(0), func),
            thrust::make_transform_iterator(
                    thrust::make_counting_iterator(n_total), func),
            begin,
            [] __device__(const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f,
                                              Eigen::Vector3f> &x) {
                const Eigen::Vector3f &pt = thrust::get<0>(x);
                return !(isnan(pt(0)) || isnan(pt(1)) || isnan(pt(2)));
            });
    resize_all(thrust::distance(begin, end_p), pointcloud->points_,
               pointcloud->normals_, pointcloud->colors_);
    if (color_type_ == TSDFVolumeColorType::NoColor)
        pointcloud->colors_.clear();
    return pointcloud;
}

std::shared_ptr<geometry::TriangleMesh>
UniformTSDFVolume::ExtractTriangleMesh() {
    // implementation of marching cubes, based on
    // http://paulbourke.net/geometry/polygonise/
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    size_t n_valid_voxels = thrust::count_if(
            enumerate_begin(voxels_), enumerate_end(voxels_),
            count_valid_voxels_functor(thrust::raw_pointer_cast(voxels_.data()),
                                       resolution_));
    size_t res3 = (resolution_ - 1) * (resolution_ - 1) * (resolution_ - 1);

    // compute cube indices for each voxels
    utility::device_vector<Eigen::Vector3i> keys(n_valid_voxels);
    utility::device_vector<int> cube_indices(n_valid_voxels);
    extract_mesh_phase0_functor func0(thrust::raw_pointer_cast(voxels_.data()),
                                      resolution_);
    thrust::copy_if(
            thrust::make_transform_iterator(
                    thrust::make_counting_iterator<size_t>(0), func0),
            thrust::make_transform_iterator(
                    thrust::make_counting_iterator(res3), func0),
            make_tuple_begin(keys, cube_indices),
            [] __device__(const thrust::tuple<Eigen::Vector3i, int> &x) {
                return thrust::get<1>(x) >= 0;
            });
    auto check_fn =
            [] __device__(
                    const thrust::tuple<Eigen::Vector3i, int> &x) -> bool {
        int cidx = thrust::get<1>(x);
        return (cidx <= 0 || cidx >= 255);
    };
    size_t n_result1 = remove_if_vectors(utility::exec_policy(0)->on(0),
                                         check_fn, keys, cube_indices);

    utility::device_vector<float> fs(n_result1 * 8);
    utility::device_vector<Eigen::Vector3f> cs(n_result1 * 8);
    extract_mesh_phase1_functor func1(thrust::raw_pointer_cast(voxels_.data()),
                                      thrust::raw_pointer_cast(keys.data()),
                                      resolution_, color_type_);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_result1 * 8),
                      make_tuple_begin(fs, cs), func1);

    // compute vertices and vertex_colors
    int *ci_p = thrust::raw_pointer_cast(cube_indices.data());
    size_t n_valid_cubes =
            thrust::count_if(thrust::make_counting_iterator<size_t>(0),
                             thrust::make_counting_iterator(n_result1 * 12),
                             [ci_p] __device__(size_t idx) {
                                 int i = idx / 12;
                                 int j = idx % 12;
                                 return (edge_table[ci_p[i]] & (1 << j)) > 0;
                             });
    resize_all(n_valid_cubes, mesh->vertices_, mesh->vertex_colors_);
    utility::device_vector<Eigen::Vector3i> repeat_keys(n_valid_cubes);
    utility::device_vector<int> repeat_cube_indices(n_valid_cubes);
    utility::device_vector<int> vert_no(n_valid_cubes);
    extract_mesh_phase2_functor func2(
            thrust::raw_pointer_cast(keys.data()),
            thrust::raw_pointer_cast(cube_indices.data()), origin_, resolution_,
            voxel_length_, thrust::raw_pointer_cast(fs.data()),
            thrust::raw_pointer_cast(cs.data()), color_type_);
    thrust::copy_if(
            thrust::make_transform_iterator(
                    thrust::make_counting_iterator<size_t>(0), func2),
            thrust::make_transform_iterator(
                    thrust::make_counting_iterator(n_result1 * 12), func2),
            make_tuple_begin(repeat_keys, repeat_cube_indices, vert_no,
                             mesh->vertices_, mesh->vertex_colors_),
            [] __device__(
                    const thrust::tuple<Eigen::Vector3i, int, int,
                                        Eigen::Vector3f, Eigen::Vector3f> &x) {
                return thrust::get<0>(x)[0] >= 0;
            });

    // compute triangles
    utility::device_vector<int> vt_offsets(n_valid_cubes + 1, 0);
    auto end2 = thrust::reduce_by_key(
            utility::exec_policy(0)->on(0), repeat_keys.begin(),
            repeat_keys.end(), thrust::make_constant_iterator<int>(1),
            thrust::make_discard_iterator(), vt_offsets.begin());
    size_t n_result2 = thrust::distance(vt_offsets.begin(), end2.second);
    vt_offsets.resize(n_result2 + 1);
    thrust::exclusive_scan(utility::exec_policy(0)->on(0), vt_offsets.begin(),
                           vt_offsets.end(), vt_offsets.begin());
    mesh->triangles_.resize(n_result2 * 4, Eigen::Vector3i(-1, -1, -1));
    extract_mesh_phase3_functor func3(
            thrust::raw_pointer_cast(repeat_cube_indices.data()),
            thrust::raw_pointer_cast(vert_no.data()),
            thrust::raw_pointer_cast(vt_offsets.data()),
            thrust::raw_pointer_cast(mesh->triangles_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(n_result2), func3);
    auto end3 = thrust::remove_if(
            utility::exec_policy(0)->on(0), mesh->triangles_.begin(),
            mesh->triangles_.end(),
            [] __device__(const Eigen::Vector3i &idxs) { return idxs[0] < 0; });
    mesh->triangles_.resize(thrust::distance(mesh->triangles_.begin(), end3));
    return mesh;
}

std::shared_ptr<geometry::PointCloud>
UniformTSDFVolume::ExtractVoxelPointCloud() const {
    auto voxel = std::make_shared<geometry::PointCloud>();
    // const float *p_tsdf = (const float *)tsdf_.data();
    // const float *p_weight = (const float *)weight_.data();
    // const float *p_color = (const float *)color_.data();
    size_t n_valid_voxels =
            thrust::count_if(voxels_.begin(), voxels_.end(),
                             [] __device__(const geometry::TSDFVoxel &v) {
                                 return (v.weight_ != 0.0f && v.tsdf_ < 0.98f &&
                                         v.tsdf_ >= -0.98f);
                             });
    extract_voxel_pointcloud_functor func(origin_, resolution_, voxel_length_);
    resize_all(n_valid_voxels, voxel->points_, voxel->colors_);
    thrust::copy_if(
            thrust::make_transform_iterator(enumerate_begin(voxels_), func),
            thrust::make_transform_iterator(enumerate_end(voxels_), func),
            make_tuple_begin(voxel->points_, voxel->colors_),
            [] __device__(
                    const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> &x) {
                const Eigen::Vector3f &pt = thrust::get<0>(x);
                return !(isnan(pt(0)) || isnan(pt(1)) || isnan(pt(2)));
            });
    voxel->RemoveNoneFinitePoints(true, false);
    return voxel;
}

std::shared_ptr<geometry::VoxelGrid> UniformTSDFVolume::ExtractVoxelGrid()
        const {
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid->voxel_size_ = voxel_length_;
    voxel_grid->origin_ = origin_ - Eigen::Vector3f::Constant(resolution_ / 2) *
                                            voxel_length_;
    size_t n_valid_voxels = thrust::count_if(
            utility::exec_policy(0)->on(0), voxels_.begin(), voxels_.end(),
            [] __device__(const geometry::TSDFVoxel &v) {
                return (v.weight_ != 0.0f && v.tsdf_ < 0.98f &&
                        v.tsdf_ >= -0.98f);
            });
    resize_all(n_valid_voxels, voxel_grid->voxels_keys_,
               voxel_grid->voxels_values_);
    extract_voxel_grid_functor func(resolution_);
    thrust::copy_if(
            thrust::make_transform_iterator(enumerate_begin(voxels_), func),
            thrust::make_transform_iterator(enumerate_end(voxels_), func),
            make_tuple_begin(voxel_grid->voxels_keys_,
                             voxel_grid->voxels_values_),
            [] __device__(
                    const thrust::tuple<Eigen::Vector3i, geometry::Voxel> &x) {
                return thrust::get<0>(x) !=
                       Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX,
                                       geometry::INVALID_VOXEL_INDEX,
                                       geometry::INVALID_VOXEL_INDEX);
            });
    return voxel_grid;
}

void UniformTSDFVolume::IntegrateWithDepthToCameraDistanceMultiplier(
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
    voxels_.resize(voxel_num_);
    uniform_integrate_functor func(
            fx, fy, cx, cy, extrinsic, voxel_length_, sdf_trunc_, safe_width,
            safe_height, resolution_,
            thrust::raw_pointer_cast(image.color_.data_.data()),
            thrust::raw_pointer_cast(image.depth_.data_.data()),
            thrust::raw_pointer_cast(
                    depth_to_camera_distance_multiplier.data_.data()),
            image.depth_.width_, image.color_.num_of_channels_, color_type_,
            origin_, thrust::raw_pointer_cast(voxels_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(
                             resolution_ * resolution_ * resolution_),
                     func);
}

std::shared_ptr<geometry::PointCloud> UniformTSDFVolume::Raycast(
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic,
        float sdf_trunc,
        bool project_valid_depth_only) const {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    size_t n_total = intrinsic.width_ * intrinsic.height_;
    const float fx = intrinsic.GetFocalLength().first;
    const float fy = intrinsic.GetFocalLength().second;
    const float cx = intrinsic.GetPrincipalPoint().first;
    const float cy = intrinsic.GetPrincipalPoint().second;
    pointcloud->points_.resize(n_total);
    pointcloud->normals_.resize(n_total);
    pointcloud->colors_.resize(n_total);
    raycast_tsdf_functor func(
            thrust::raw_pointer_cast(voxels_.data()), intrinsic.width_, fx, fy,
            cx, cy, utility::InverseTransform(extrinsic), voxel_length_,
            resolution_, origin_, sdf_trunc, color_type_);
    thrust::transform(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator(n_total),
            make_tuple_begin(pointcloud->points_, pointcloud->normals_,
                             pointcloud->colors_),
            func);
    pointcloud->RemoveNoneFinitePoints(project_valid_depth_only,
                                       project_valid_depth_only);
    return pointcloud;
}

}  // namespace integration
}  // namespace cupoch