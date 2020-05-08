#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/integration/marching_cubes_const.h"
#include "cupoch/integration/uniform_tsdfvolume.h"
#include "cupoch/utility/helper.h"

#include <thrust/iterator/discard_iterator.h>

using namespace cupoch;
using namespace cupoch::integration;

namespace {

__device__ float GetTSDFAt(const Eigen::Vector3f &p,
                           const geometry::TSDFVoxel *voxels,
                           float voxel_length,
                           int resolution) {
    Eigen::Vector3i idx;
    Eigen::Vector3f p_grid = p / voxel_length - Eigen::Vector3f(0.5, 0.5, 0.5);
    for (int i = 0; i < 3; i++) {
        idx(i) = (int)std::floor(p_grid(i));
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
    extract_pointcloud_functor(const geometry::TSDFVoxel* voxels,
                               int resolution,
                               float voxel_length,
                               const Eigen::Vector3f &origin,
                               TSDFVolumeColorType color_type)
        : voxels_(voxels), resolution_(resolution),
          voxel_length_(voxel_length),
          origin_(origin),
          half_voxel_length_(0.5 * voxel_length_),
          color_type_(color_type){};
    const geometry::TSDFVoxel* voxels_;
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

        Eigen::Vector3f point(std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN());
        Eigen::Vector3f normal(std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::quiet_NaN());
        Eigen::Vector3f color(std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN());
        Eigen::Vector3i idx0(x, y, z);
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
        return thrust::make_tuple(point, normal, color);
    }
};

struct count_valid_voxels_functor {
    count_valid_voxels_functor(const geometry::TSDFVoxel* voxels, int resolution)
    : voxels_(voxels), resolution_(resolution) {};
    const geometry::TSDFVoxel* voxels_;
    const int resolution_;
    __device__ bool operator() (const geometry::TSDFVoxel& v) const {
        if (v.grid_index_[0] == resolution_ - 1 ||
            v.grid_index_[1] == resolution_ - 1 ||
            v.grid_index_[2] == resolution_ - 1)
            return false;
        for (int i = 0; i < 8; ++i) {
           Eigen::Vector3i idx = v.grid_index_ + Eigen::Vector3i(shift[i][0], shift[i][1], shift[i][2]);
           if (voxels_[IndexOf(idx, resolution_)].weight_ == 0.0f) return false;
        }
        return true;
    }
};

struct extract_mesh_phase0_functor {
    extract_mesh_phase0_functor(const geometry::TSDFVoxel *voxels,
                                int resolution)
        : voxels_(voxels), resolution_(resolution) {};
    const geometry::TSDFVoxel *voxels_;
    const int resolution_;
    __device__ thrust::tuple<Eigen::Vector3i, int> operator()(size_t idx) {
        int res2 = (resolution_ - 1) * (resolution_ - 1);
        int x = idx / res2;
        int yz = idx % res2;
        int y = yz / (resolution_ - 1);
        int z = yz % (resolution_ - 1);

        int cube_index = 0;
        Eigen::Vector3i key = Eigen::Vector3i(x, y, z);
        for (int i = 0; i < 8; ++i) {
            Eigen::Vector3i idxs =
                    key + Eigen::Vector3i(shift[i][0], shift[i][1], shift[i][2]);
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
        : voxels_(voxels), keys_(keys),
        resolution_(resolution),
        color_type_(color_type) {};
    const geometry::TSDFVoxel *voxels_;
    const Eigen::Vector3i* keys_;
    const int resolution_;
    TSDFVolumeColorType color_type_;
    __device__ thrust::tuple<float, Eigen::Vector3f>
    operator()(size_t idx) {
        int j = idx / 8;
        int i = idx % 8;
        const Eigen::Vector3i& key = keys_[j];
        Eigen::Vector3i idxs =
                key + Eigen::Vector3i(shift[i][0], shift[i][1], shift[i][2]);
        Eigen::Vector3f c = Eigen::Vector3f::Zero();
        if (voxels_[IndexOf(idxs, resolution_)].weight_ == 0.0f) {
            return thrust::make_tuple(0.0f, c);
        } else {
            float f = voxels_[IndexOf(idxs, resolution_)].tsdf_;
            if (color_type_ == TSDFVolumeColorType::RGB8) {
                c = voxels_[IndexOf(idxs, resolution_)].color_ / 255.0;
            } else if (color_type_ == TSDFVolumeColorType::Gray32) {
                c = voxels_[IndexOf(idxs, resolution_)].color_;
            }
            return thrust::make_tuple(f, c);
        }
    }
};

struct extract_mesh_phase2_functor {
    extract_mesh_phase2_functor(const Eigen::Vector3i* keys,
                                const int* cube_indices,
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
    const Eigen::Vector3i* keys_;
    const int* cube_indices_;
    const Eigen::Vector3f origin_;
    const int resolution_;
    const float voxel_length_;
    const float half_voxel_length_;
    const float *fs_;
    const Eigen::Vector3f *cs_;
    const TSDFVolumeColorType color_type_;
    __device__ thrust::tuple<Eigen::Vector3i,
                             int,
                             int,
                             Eigen::Vector3f,
                             Eigen::Vector3f>
    operator() (size_t idx) const {
        int j = idx / 12;
        const Eigen::Vector3i& xyz = keys_[j];
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
            return thrust::make_tuple(xyz, cube_index, i, vertex, vertex_color);
        } else {
            Eigen::Vector3i index = -Eigen::Vector3i::Ones();
            Eigen::Vector3f vertex = Eigen::Vector3f::Zero();
            Eigen::Vector3f vertex_color = Eigen::Vector3f::Zero();
            return thrust::make_tuple(index, cube_index, i, vertex, vertex_color);
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
          triangles_(triangles) {};
    const int *cube_index_;
    const int *vert_no_;
    const int *key_index_;
    Eigen::Vector3i *triangles_;
    __device__ void operator()(size_t idx) {
        for (int j = key_index_[idx]; j < key_index_[idx + 1]; ++j) {
            for (int i = 0; tri_table[cube_index_[j]][i] != -1; ++i) {
                int tri_idx = tri_table[cube_index_[j]][i];
                for (int l = key_index_[idx]; l < key_index_[idx + 1]; ++l) {
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
            const geometry::TSDFVoxel& v) {
        int x = v.grid_index_[0];
        int y = v.grid_index_[1];
        int z = v.grid_index_[2];
        Eigen::Vector3f pt(half_voxel_length_ + voxel_length_ * x,
                           half_voxel_length_ + voxel_length_ * y,
                           half_voxel_length_ + voxel_length_ * z);
        if (v.weight_ != 0.0f && v.tsdf_ < 0.98f && v.tsdf_ >= -0.98f) {
            float c = (v.tsdf_ + 1.0) * 0.5;
            return thrust::make_tuple(pt + origin_, Eigen::Vector3f(c, c, c));
        }
        return thrust::make_tuple(
                Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN()),
                Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN()));
    }
};

struct extract_voxel_grid_functor {
    extract_voxel_grid_functor(int resolution)
        : resolution_(resolution){};
    const int resolution_;
    __device__ thrust::tuple<Eigen::Vector3i, geometry::Voxel> operator()(
            const geometry::TSDFVoxel& v) {
        const float w = v.weight_;
        const float f = v.tsdf_;
        if (w != 0.0f && f < 0.98f && f >= -0.98f) {
            float c = (f + 1.0) * 0.5;
            return thrust::make_tuple(v.grid_index_,
                                      geometry::Voxel(v.grid_index_, Eigen::Vector3f(c, c, c)));
        }
        return thrust::make_tuple(Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX,
                                                  geometry::INVALID_VOXEL_INDEX,
                                                  geometry::INVALID_VOXEL_INDEX),
                                  geometry::Voxel());
    }
};

struct integrate_functor {
    integrate_functor(const Eigen::Vector3f &origin,
                      float fx,
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
                      geometry::TSDFVoxel *voxels)
        : origin_(origin),
          fx_(fx),
          fy_(fy),
          cx_(cx),
          cy_(cy),
          extrinsic_(extrinsic),
          voxel_length_(voxel_length),
          half_voxel_length_(0.5 * voxel_length),
          sdf_trunc_(sdf_trunc),
          sdf_trunc_inv_(1.0 / sdf_trunc),
          extrinsic_scaled_(voxel_length * extrinsic),
          safe_width_(safe_width),
          safe_height_(safe_height),
          resolution_(resolution),
          color_(color),
          depth_(depth),
          depth_to_camera_distance_multiplier_(
                  depth_to_camera_distance_multiplier),
          width_(width),
          num_of_channels_(num_of_channels),
          color_type_(color_type),
          voxels_(voxels){};
    const Eigen::Vector3f origin_;
    const float fx_;
    const float fy_;
    const float cx_;
    const float cy_;
    const Eigen::Matrix4f extrinsic_;
    const float voxel_length_;
    const float half_voxel_length_;
    const float sdf_trunc_;
    const float sdf_trunc_inv_;
    const Eigen::Matrix4f extrinsic_scaled_;
    const float safe_width_;
    const float safe_height_;
    const int resolution_;
    const uint8_t *color_;
    const uint8_t *depth_;
    const uint8_t *depth_to_camera_distance_multiplier_;
    const int width_;
    const int num_of_channels_;
    const TSDFVolumeColorType color_type_;
    geometry::TSDFVoxel *voxels_;
    __device__ void operator()(size_t idx) {
        int res2 = resolution_ * resolution_;
        int x = idx / res2;
        int yz = idx % res2;
        int y = yz / resolution_;
        int z = yz % resolution_;
        voxels_[idx].grid_index_ = Eigen::Vector3i(x, y, z);

        Eigen::Vector4f pt_3d_homo(
                float(half_voxel_length_ + voxel_length_ * x + origin_(0)),
                float(half_voxel_length_ + voxel_length_ * y + origin_(1)),
                float(half_voxel_length_ + origin_(2)), 1.f);
        Eigen::Vector4f pt_camera = extrinsic_ * pt_3d_homo;
        pt_camera(0) += z * extrinsic_scaled_(0, 2);
        pt_camera(1) += z * extrinsic_scaled_(1, 2);
        pt_camera(2) += z * extrinsic_scaled_(2, 2);
        // Skip if negative depth after projection
        if (pt_camera(2) <= 0) {
            return;
        }
        // Skip if x-y coordinate not in range
        float u_f = pt_camera(0) * fx_ / pt_camera(2) + cx_ + 0.5f;
        float v_f = pt_camera(1) * fy_ / pt_camera(2) + cy_ + 0.5f;
        if (!(u_f >= 0.0001f && u_f < safe_width_ && v_f >= 0.0001f &&
              v_f < safe_height_)) {
            return;
        }
        // Skip if negative depth in depth image
        int u = (int)u_f;
        int v = (int)v_f;
        float d = *geometry::PointerAt<float>(depth_, width_, u, v);
        if (d <= 0.0f) {
            return;
        }

        float sdf =
                (d - pt_camera(2)) *
                (*geometry::PointerAt<float>(
                        depth_to_camera_distance_multiplier_, width_, u, v));
        if (sdf > -sdf_trunc_) {
            // integrate
            float tsdf = min(1.0f, sdf * sdf_trunc_inv_);
            voxels_[idx].tsdf_ =
                    (voxels_[idx].tsdf_ * voxels_[idx].weight_ + tsdf) /
                    (voxels_[idx].weight_ + 1.0f);
            if (color_type_ == TSDFVolumeColorType::RGB8) {
                const uint8_t *rgb = geometry::PointerAt<uint8_t>(
                        color_, width_, num_of_channels_, u, v, 0);
                Eigen::Vector3f rgb_f(rgb[0], rgb[1], rgb[2]);
                voxels_[idx].color_ =
                        (voxels_[idx].color_ * voxels_[idx].weight_ +
                         rgb_f) /
                        (voxels_[idx].weight_ + 1.0f);
            } else if (color_type_ == TSDFVolumeColorType::Gray32) {
                const float *intensity = geometry::PointerAt<float>(
                        color_, width_, num_of_channels_, u, v, 0);
                voxels_[idx].color_ = (voxels_[idx].color_.array() *
                                                voxels_[idx].weight_ +
                                         (*intensity)) /
                                        (voxels_[idx].weight_ + 1.0f);
            }
            voxels_[idx].weight_ += 1.0f;
        }
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
 : TSDFVolume(other), voxels_(other.voxels_), origin_(other.origin_),
 length_(other.length_), resolution_(other.resolution_), voxel_num_(other.voxel_num_)
{}

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
    size_t n_valid_voxels = thrust::count_if(voxels_.begin(), voxels_.end(),
                                             [] __device__ (const geometry::TSDFVoxel& v) {
                                                 return (v.weight_ != 0.0f && v.tsdf_ < 0.98f && v.tsdf_ >= -0.98f);
                                             });
    extract_pointcloud_functor func(thrust::raw_pointer_cast(voxels_.data()),
                                    resolution_, voxel_length_, origin_,
                                    color_type_);
    pointcloud->points_.resize(n_valid_voxels);
    pointcloud->normals_.resize(n_valid_voxels);
    pointcloud->colors_.resize(n_valid_voxels);
    size_t n_total = (resolution_ - 2) * (resolution_ - 2) * (resolution_ - 2) * 3;
    auto begin = make_tuple_iterator(pointcloud->points_.begin(),
                                     pointcloud->normals_.begin(),
                                     pointcloud->colors_.begin());
    auto end_p = thrust::copy_if(thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0), func),
                                 thrust::make_transform_iterator(thrust::make_counting_iterator(n_total), func),
                                 begin,
                                 [] __device__ (const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>& x) {
                                     const Eigen::Vector3f& pt = thrust::get<0>(x);
                                     return !(isnan(pt(0)) || isnan(pt(1)) || isnan(pt(2)));
                                 });
    resize_all(thrust::distance(begin, end_p), pointcloud->points_, pointcloud->normals_, pointcloud->colors_);
    if (color_type_ == TSDFVolumeColorType::NoColor) pointcloud->colors_.clear();
    return pointcloud;
}

std::shared_ptr<geometry::TriangleMesh>
UniformTSDFVolume::ExtractTriangleMesh() {
    // implementation of marching cubes, based on
    // http://paulbourke.net/geometry/polygonise/
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    size_t n_valid_voxels = thrust::count_if(voxels_.begin(), voxels_.end(),
            count_valid_voxels_functor(thrust::raw_pointer_cast(voxels_.data()),
                                                                resolution_));
    size_t res3 = (resolution_ - 1) * (resolution_ - 1) * (resolution_ - 1);

    // compute cube indices for each voxels
    utility::device_vector<Eigen::Vector3i> keys(n_valid_voxels);
    utility::device_vector<int> cube_indices(n_valid_voxels);
    extract_mesh_phase0_functor func0(thrust::raw_pointer_cast(voxels_.data()),
                                      resolution_);
    thrust::copy_if(
            thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0), func0),
            thrust::make_transform_iterator(thrust::make_counting_iterator(res3), func0),
            make_tuple_iterator(keys.begin(), cube_indices.begin()),
            [] __device__ (const thrust::tuple<Eigen::Vector3i, int>& x) {
                return thrust::get<1>(x) >= 0;
            });
    auto begin1 = make_tuple_iterator(keys.begin(), cube_indices.begin());
    auto end1 = thrust::remove_if(
            begin1,
            make_tuple_iterator(keys.end(), cube_indices.end()),
            [] __device__(
                    const thrust::tuple<Eigen::Vector3i, int> &x) -> bool {
                int cidx = thrust::get<1>(x);
                return (cidx <= 0 || cidx >= 255);
            });
    size_t n_result1 = thrust::distance(begin1, end1);
    resize_all(n_result1, keys, cube_indices);

    utility::device_vector<float> fs(n_result1 * 8);
    utility::device_vector<Eigen::Vector3f> cs(n_result1 * 8);
    extract_mesh_phase1_functor func1(thrust::raw_pointer_cast(voxels_.data()),
                                      thrust::raw_pointer_cast(keys.data()),
                                      resolution_, color_type_);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_result1 * 8),
                      make_tuple_iterator(fs.begin(), cs.begin()), func1);

    // compute vertices and vertex_colors
    int* ci_p = thrust::raw_pointer_cast(cube_indices.data());
    size_t n_valid_cubes = thrust::count_if(thrust::make_counting_iterator<size_t>(0),
                                            thrust::make_counting_iterator(n_result1 * 12),
                                            [ci_p] __device__ (size_t idx) {
                                                int i = idx / 12;
                                                int j = idx % 12;
                                                return (edge_table[ci_p[i]] & (1 << j)) > 0;
                                            });
    resize_all(n_valid_cubes, mesh->vertices_, mesh->vertex_colors_);
    utility::device_vector<Eigen::Vector3i> repeat_keys(n_valid_cubes);
    utility::device_vector<int> repeat_cube_indices(n_valid_cubes);
    utility::device_vector<int> vert_no(n_valid_cubes);
    extract_mesh_phase2_functor func2(thrust::raw_pointer_cast(keys.data()),
                                      thrust::raw_pointer_cast(cube_indices.data()),
                                      origin_, voxel_length_, resolution_,
                                      thrust::raw_pointer_cast(fs.data()),
                                      thrust::raw_pointer_cast(cs.data()),
                                      color_type_);
    thrust::copy_if(
            thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0), func2),
            thrust::make_transform_iterator(thrust::make_counting_iterator(n_result1 * 12), func2),
            make_tuple_iterator(repeat_keys.begin(),
                                repeat_cube_indices.begin(),
                                vert_no.begin(),
                                mesh->vertices_.begin(),
                                mesh->vertex_colors_.begin()),
            [] __device__ (const thrust::tuple<Eigen::Vector3i, int, int, Eigen::Vector3f, Eigen::Vector3f>& x) {
                return thrust::get<0>(x)[0] >= 0;
            });


    // compute triangles
    utility::device_vector<int> vt_offsets(n_valid_cubes + 1, 0);
    auto end2 = thrust::reduce_by_key(repeat_keys.begin(), repeat_keys.end(),
                                      thrust::make_constant_iterator<int>(1),
                                      thrust::make_discard_iterator(), vt_offsets.begin());
    size_t n_result2 = thrust::distance(vt_offsets.begin(), end2.second);
    vt_offsets.resize(n_result2 + 1);
    thrust::exclusive_scan(vt_offsets.begin(), vt_offsets.end(), vt_offsets.begin());
    mesh->triangles_.resize(n_result2 * 4, Eigen::Vector3i(-1, -1, -1));
    extract_mesh_phase3_functor func3(
            thrust::raw_pointer_cast(repeat_cube_indices.data()),
            thrust::raw_pointer_cast(vert_no.data()),
            thrust::raw_pointer_cast(vt_offsets.data()),
            thrust::raw_pointer_cast(mesh->triangles_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(n_result2), func3);
    auto end3 = thrust::remove_if(
            mesh->triangles_.begin(), mesh->triangles_.end(),
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
    size_t n_valid_voxels = thrust::count_if(voxels_.begin(), voxels_.end(),
                                             [] __device__ (const geometry::TSDFVoxel& v) {
                                                 return (v.weight_ != 0.0f && v.tsdf_ < 0.98f && v.tsdf_ >= -0.98f);
                                             });
    extract_voxel_pointcloud_functor func(origin_, resolution_, voxel_length_);
    resize_all(n_valid_voxels, voxel->points_, voxel->colors_);
    thrust::copy_if(
            thrust::make_transform_iterator(voxels_.begin(), func),
            thrust::make_transform_iterator(voxels_.end(), func),
            make_tuple_iterator(voxel->points_.begin(), voxel->colors_.begin()),
            [] __device__ (const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f>& x) {
                const Eigen::Vector3f& pt = thrust::get<0>(x);
                return !(isnan(pt(0)) || isnan(pt(1)) || isnan(pt(2)));
            });
    voxel->RemoveNoneFinitePoints(true, false);
    return voxel;
}

std::shared_ptr<geometry::VoxelGrid> UniformTSDFVolume::ExtractVoxelGrid()
        const {
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid->voxel_size_ = voxel_length_;
    voxel_grid->origin_ = origin_;
    size_t n_valid_voxels = thrust::count_if(voxels_.begin(), voxels_.end(),
                                             [] __device__ (const geometry::TSDFVoxel& v) {
                                                 return (v.weight_ != 0.0f && v.tsdf_ < 0.98f && v.tsdf_ >= -0.98f);
                                             });
    resize_all(n_valid_voxels, voxel_grid->voxels_keys_, voxel_grid->voxels_values_);
    extract_voxel_grid_functor func(resolution_);
    thrust::copy_if(thrust::make_transform_iterator(voxels_.begin(), func),
                    thrust::make_transform_iterator(voxels_.end(), func),
                    make_tuple_iterator(voxel_grid->voxels_keys_.begin(),
                                        voxel_grid->voxels_values_.begin()),
                    [] __device__ (const thrust::tuple<Eigen::Vector3i, geometry::Voxel>& x) {
                        return thrust::get<0>(x) != Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX,
                            geometry::INVALID_VOXEL_INDEX, geometry::INVALID_VOXEL_INDEX);
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
    integrate_functor func(
            origin_, fx, fy, cx, cy, extrinsic, voxel_length_, sdf_trunc_,
            safe_width, safe_height, resolution_,
            thrust::raw_pointer_cast(image.color_.data_.data()),
            thrust::raw_pointer_cast(image.depth_.data_.data()),
            thrust::raw_pointer_cast(
                    depth_to_camera_distance_multiplier.data_.data()),
            image.depth_.width_, image.color_.num_of_channels_, color_type_,
            thrust::raw_pointer_cast(voxels_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(
                             resolution_ * resolution_ * resolution_),
                     func);
}