#include "cupoch/integration/uniform_tsdfvolume.h"

#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/integration/marching_cubes_const.h"
#include "cupoch/utility/range.h"
#include "cupoch/utility/helper.h"

using namespace cupoch;
using namespace cupoch::integration;

namespace {

__device__
int IndexOf(int x, int y, int z, int resolution) {
    return x * resolution * resolution + y * resolution + z;
}

__device__
int IndexOf(const Eigen::Vector3i &xyz, int resolution) {
    return IndexOf(xyz(0), xyz(1), xyz(2), resolution);
}

__device__
float GetTSDFAt(const Eigen::Vector3f &p, const geometry::TSDFVoxel* voxels,
                float voxel_length, int resolution) {
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

__device__
Eigen::Vector3f GetNormalAt(const Eigen::Vector3f &p, const geometry::TSDFVoxel* voxels,
                            float voxel_length, int resolution) {
    Eigen::Vector3f n;
    const double half_gap = 0.99 * voxel_length;
    for (int i = 0; i < 3; i++) {
        Eigen::Vector3f p0 = p;
        p0(i) -= half_gap;
        Eigen::Vector3f p1 = p;
        p1(i) += half_gap;
        n(i) = GetTSDFAt(p1, voxels, voxel_length, resolution) - GetTSDFAt(p0, voxels, voxel_length, resolution);
    }
    return n.normalized();
}

struct extract_pointcloud_functor {
    extract_pointcloud_functor(const geometry::TSDFVoxel* voxels,
                               int resolution, float voxel_length,
                               const Eigen::Vector3f& origin,
                               TSDFVolumeColorType color_type)
     : voxels_(voxels), resolution_(resolution), voxel_length_(voxel_length),
       origin_(origin), half_voxel_length_(0.5 * voxel_length_),
       color_type_(color_type) {};
    const geometry::TSDFVoxel* voxels_;
    const int resolution_;
    const float voxel_length_;
    const Eigen::Vector3f origin_;
    const float half_voxel_length_;
    const TSDFVolumeColorType color_type_;
    __device__
    thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f> operator() (size_t idx) {
        int res2 = (resolution_ - 1) * (resolution_ - 1);
        int x = idx / (3 * res2);
        int yzi = idx % (3 * res2);
        int y = yzi / (3 * (resolution_ - 1));
        int zi = yzi % (3 * (resolution_ - 1));
        int z = zi / 3;
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
        float w0 = voxels_[idx].weight_;
        float f0 = voxels_[idx].tsdf_;
        const Eigen::Vector3f &c0 = voxels_[idx].color_;
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
            if (w1 != 0.0f && f1 < 0.98f && f1 >= -0.98f &&
                f0 * f1 < 0) {
                float r0 = std::fabs(f0);
                float r1 = std::fabs(f1);
                Eigen::Vector3f p = p0;
                p(i) = (p0(i) * r1 + p1(i) * r0) / (r0 + r1);
                point = p + origin_;
                if (color_type_ == TSDFVolumeColorType::RGB8) {
                    color = (c0 * r1 + c1 * r0) / (r0 + r1) / 255.0f;
                } else if (color_type_ ==
                           TSDFVolumeColorType::Gray32) {
                    color = (c0 * r1 + c1 * r0) / (r0 + r1);
                }
                // has_normal
                normal = GetNormalAt(p, voxels_, voxel_length_, resolution_);
            }
        }
        return thrust::make_tuple(point, normal, color);
    }
};

struct extract_mesh_phase1_functor {
    extract_mesh_phase1_functor(const geometry::TSDFVoxel* voxels, int resolution,
                                TSDFVolumeColorType color_type)
                                : voxels_(voxels), resolution_(resolution), color_type_(color_type) {};
    const geometry::TSDFVoxel* voxels_;
    const int resolution_;
    TSDFVolumeColorType color_type_;
    __device__
    thrust::tuple<Eigen::Vector3i, int, float, Eigen::Vector3f> operator() (size_t idx) {
        int res2 = (resolution_ - 1) * (resolution_ - 1);
        int x = idx / (8 * res2);
        int yzi = idx % (8 * res2);
        int y = yzi / (8 * (resolution_ - 1));
        int zi = yzi % (8 * (resolution_ - 1));
        int z = zi / 8;
        int i = zi % 8;

        Eigen::Vector3i key = Eigen::Vector3i(x, y, z);
        Eigen::Vector3i idxs = key + Eigen::Vector3i(shift[i][0], shift[i][1], shift[i][2]);
        Eigen::Vector3f c = Eigen::Vector3f::Zero();
        if (voxels_[IndexOf(idxs, resolution_)].weight_ == 0.0f) {
            return thrust::make_tuple(key, -1, 0.0f, c);
        } else {
            int cube_index = 0;
            float f = voxels_[IndexOf(idxs, resolution_)].tsdf_;
            if (f < 0.0f) {
                cube_index = (1 << i);
            }
            if (color_type_ == TSDFVolumeColorType::RGB8) {
                c = voxels_[IndexOf(idxs, resolution_)].color_.cast<float>() /
                       255.0;
            } else if (color_type_ == TSDFVolumeColorType::Gray32) {
                c = voxels_[IndexOf(idxs, resolution_)].color_.cast<float>();
            }
            return thrust::make_tuple(key, cube_index, f, c);
        }
    }
};

struct add_cube_index_functor {
    __device__
    int operator() (int x, int y) {
        return (x < 0 || y < 0) ? -1 : x + y;
    }
};

struct extract_mesh_phase2_functor {
    extract_mesh_phase2_functor(const Eigen::Vector3f& origin,
                                int resolution,
                                float voxel_length,
                                const float* fs,
                                const Eigen::Vector3f* cs,
                                TSDFVolumeColorType color_type)
     : origin_(origin), resolution_(resolution), voxel_length_(voxel_length),
       half_voxel_length_(0.5 * voxel_length_), fs_(fs), cs_(cs), color_type_(color_type) {};
    const Eigen::Vector3f origin_;
    const int resolution_;
    const float voxel_length_;
    const float half_voxel_length_;
    const float* fs_;
    const Eigen::Vector3f* cs_;
    const TSDFVolumeColorType color_type_;
    __device__
    thrust::tuple<Eigen::Vector4i, Eigen::Vector3i, int, int, Eigen::Vector3f, Eigen::Vector3f> operator() (const thrust::tuple<Eigen::Vector3i, int>& idxs, int idx) const {
        Eigen::Vector3i xyz = thrust::get<0>(idxs);
        int x = xyz[0];
        int y = xyz[1];
        int z = xyz[2];
        int offset = IndexOf(xyz, resolution_);
        int cube_index = thrust::get<1>(idxs);
        int i = idx % 12;
        if (edge_table[cube_index] & (1 << i)) {
            Eigen::Vector4i edge_index =
                    Eigen::Vector4i(x, y, z, 0) + Eigen::Vector4i(edge_shift[i][0], edge_shift[i][1],
                                                                  edge_shift[i][2], edge_shift[i][3]);
            Eigen::Vector3f pt(
                    half_voxel_length_ +
                            voxel_length_ * edge_index(0),
                    half_voxel_length_ +
                            voxel_length_ * edge_index(1),
                    half_voxel_length_ +
                            voxel_length_ * edge_index(2));
            float f0 = std::abs(fs_[offset + edge_to_vert[i][0]]);
            float f1 = std::abs(fs_[offset + edge_to_vert[i][1]]);
            pt(edge_index(3)) += f0 * voxel_length_ / (f0 + f1);
            Eigen::Vector3f vertex = pt + origin_;
            Eigen::Vector3f vertex_color = Eigen::Vector3f::Zero();
            if (color_type_ != TSDFVolumeColorType::NoColor) {
                const auto &c0 = cs_[offset + edge_to_vert[i][0]];
                const auto &c1 = cs_[offset + edge_to_vert[i][1]];
                vertex_color = (f1 * c0 + f0 * c1) / (f0 + f1);
            }
            return thrust::make_tuple(edge_index, xyz, cube_index, i, vertex, vertex_color);
        } else {
            Eigen::Vector4i edge_index = -Eigen::Vector4i::Ones();
            Eigen::Vector3f vertex = Eigen::Vector3f::Zero();
            Eigen::Vector3f vertex_color = Eigen::Vector3f::Zero();
            return thrust::make_tuple(edge_index, xyz, cube_index, i, vertex, vertex_color);
        }
    }
};

struct extract_mesh_phase3_functor {
    extract_mesh_phase3_functor(const int* cube_index, const int* vert_no,
                                const int* key_index,
                                Eigen::Vector3i* triangles)
                                : cube_index_(cube_index), vert_no_(vert_no),
                                  key_index_(key_index), triangles_(triangles) {};
    const int* cube_index_;
    const int* vert_no_;
    const int* key_index_;
    Eigen::Vector3i* triangles_;
    __device__
    void operator() (size_t idx) {
        int j[12] = {0};
        int n = key_index_[idx + 1] - key_index_[idx];
        for (int i = 0; i < n; ++i) {
            int k = key_index_[idx] + i;
            int tri_idx = tri_table[cube_index_[k]][vert_no_[k]][j[vert_no_[k]]];
            if (tri_idx < 0) continue;
            triangles_[idx + tri_idx / 3][tri_idx % 3] = k;
            j[k]++;
        }
    }
};

struct extract_voxel_pointcloud_functor {
    extract_voxel_pointcloud_functor(const geometry::TSDFVoxel* voxels,
                                     const Eigen::Vector3f& origin, int resolution,
                                     float voxel_length)
                                     : voxels_(voxels), origin_(origin), resolution_(resolution),
                                       voxel_length_(voxel_length), half_voxel_length_(0.5 * voxel_length) {};
    const geometry::TSDFVoxel* voxels_;
    const Eigen::Vector3f origin_;
    const int resolution_;
    const float voxel_length_;
    const float half_voxel_length_;
    __device__
    thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> operator() (size_t idx) {
        int res2 = resolution_ * resolution_;
        int x = idx / res2;
        int yz = idx % res2;
        int y = yz / resolution_;
        int z = yz % resolution_;
        Eigen::Vector3f pt(half_voxel_length_ + voxel_length_ * x,
            half_voxel_length_ + voxel_length_ * y,
            half_voxel_length_ + voxel_length_ * z);
        int ind = IndexOf(x, y, z, resolution_);
        if (voxels_[ind].weight_ != 0.0f &&
            voxels_[ind].tsdf_ < 0.98f &&
            voxels_[ind].tsdf_ >= -0.98f) {
            float c = (voxels_[ind].tsdf_ + 1.0) * 0.5;
            return thrust::make_tuple(pt + origin_, Eigen::Vector3f(c, c, c));
        }
        return thrust::make_tuple(Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),
                                                  std::numeric_limits<float>::quiet_NaN(),
                                                  std::numeric_limits<float>::quiet_NaN()),
                                  Eigen::Vector3f(std::numeric_limits<float>::quiet_NaN(),
                                                  std::numeric_limits<float>::quiet_NaN(),
                                                  std::numeric_limits<float>::quiet_NaN()));
    }
};

struct extract_voxel_grid_functor {
    extract_voxel_grid_functor(const geometry::TSDFVoxel* voxels, int resolution) : voxels_(voxels), resolution_(resolution) {};
    const geometry::TSDFVoxel* voxels_;
    const int resolution_;
    __device__
    thrust::tuple<Eigen::Vector3i, geometry::Voxel> operator() (size_t idx) {
        int res2 = resolution_ * resolution_;
        int x = idx / res2;
        int yz = idx % res2;
        int y = yz / resolution_;
        int z = yz % resolution_;

        const int ind = IndexOf(x, y, z, resolution_);
        const float w = voxels_[ind].weight_;
        const float f = voxels_[ind].tsdf_;
        if (w != 0.0f && f < 0.98f && f >= -0.98f) {
            float c = (f + 1.0) * 0.5;
            Eigen::Vector3f color = Eigen::Vector3f(c, c, c);
            Eigen::Vector3i index = Eigen::Vector3i(x, y, z);
            return thrust::make_tuple(index, geometry::Voxel(index, color));
        }
        return thrust::make_tuple(Eigen::Vector3i(-1, -1, -1), geometry::Voxel());
    }
};

struct integrate_functor {
    integrate_functor(const Eigen::Vector3f& origin, float fx, float fy, float cx, float cy,
                      const Eigen::Matrix4f& extrinsic, float voxel_length, float sdf_trunc,
                      float safe_width, float safe_height, int resolution,
                      const uint8_t* color, const uint8_t* depth,
                      const uint8_t* depth_to_camera_distance_multiplier,
                      int width, int num_of_channels, TSDFVolumeColorType color_type,
                      geometry::TSDFVoxel* voxels)
                      : origin_(origin), fx_(fx), fy_(fy), cx_(cx), cy_(cy), extrinsic_(extrinsic),
                       voxel_length_(voxel_length), half_voxel_length_(0.5 * voxel_length),
                       sdf_trunc_(sdf_trunc), sdf_trunc_inv_(1.0 / sdf_trunc), extrinsic_scaled_(voxel_length * extrinsic),
                       safe_width_(safe_width), safe_height_(safe_height), resolution_(resolution), color_(color),
                       depth_(depth), depth_to_camera_distance_multiplier_(depth_to_camera_distance_multiplier),
                       width_(width), num_of_channels_(num_of_channels),
                       color_type_(color_type), voxels_(voxels) {};
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
    const uint8_t* color_;
    const uint8_t* depth_;
    const uint8_t* depth_to_camera_distance_multiplier_;
    const int width_;
    const int num_of_channels_;
    const TSDFVolumeColorType color_type_;
    geometry::TSDFVoxel* voxels_;
    __device__
    void operator() (size_t idx) {
        int res2 = resolution_ * resolution_;
        int x = idx / res2;
        int yz = idx % res2;
        int y = yz / resolution_;
        int z = yz % resolution_;

        Eigen::Vector4f pt_3d_homo(float(half_voxel_length_ +
                                         voxel_length_ * x + origin_(0)),
                                   float(half_voxel_length_ +
                                         voxel_length_ * y + origin_(1)),
                                   float(half_voxel_length_ + origin_(2)),
                                   1.f);
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

        int v_ind = IndexOf(x, y, z, resolution_);
        float sdf =
                (d - pt_camera(2)) *
                (*geometry::PointerAt<float>(depth_to_camera_distance_multiplier_, width_,
                        u, v));
        if (sdf > -sdf_trunc_) {
            // integrate
            float tsdf = std::min(1.0f, sdf * sdf_trunc_inv_);
            voxels_[v_ind].tsdf_ =
                    (voxels_[v_ind].tsdf_ * voxels_[v_ind].weight_ +
                     tsdf) /
                    (voxels_[v_ind].weight_ + 1.0f);
            if (color_type_ == TSDFVolumeColorType::RGB8) {
                const uint8_t *rgb =
                        geometry::PointerAt<uint8_t>(color_, width_, num_of_channels_, u, v, 0);
                Eigen::Vector3f rgb_f(rgb[0], rgb[1], rgb[2]);
                voxels_[v_ind].color_ =
                        (voxels_[v_ind].color_ *
                                 voxels_[v_ind].weight_ +
                         rgb_f) /
                        (voxels_[v_ind].weight_ + 1.0f);
            } else if (color_type_ == TSDFVolumeColorType::Gray32) {
                const float *intensity =
                        geometry::PointerAt<float>(color_, width_, num_of_channels_, u, v, 0);
                voxels_[v_ind].color_ =
                        (voxels_[v_ind].color_.array() *
                                 voxels_[v_ind].weight_ +
                         (*intensity)) /
                        (voxels_[v_ind].weight_ + 1.0f);
            }
            voxels_[v_ind].weight_ += 1.0f;
        }
    }
};

}

UniformTSDFVolume::UniformTSDFVolume(
        float length,
        int resolution,
        float sdf_trunc,
        TSDFVolumeColorType color_type,
        const Eigen::Vector3f &origin /* = Eigen::Vector3d::Zero()*/)
    : TSDFVolume(length / (float)resolution, sdf_trunc, color_type),
      origin_(origin),
      length_(length),
      resolution_(resolution),
      voxel_num_(resolution * resolution * resolution) {
    voxels_.resize(voxel_num_);
    SetConstants();
}

UniformTSDFVolume::~UniformTSDFVolume() {}

void UniformTSDFVolume::Reset() { voxels_.clear(); }

std::shared_ptr<geometry::PointCloud> UniformTSDFVolume::ExtractPointCloud() {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    extract_pointcloud_functor func(thrust::raw_pointer_cast(voxels_.data()),
                                    resolution_, voxel_length_,
                                    origin_, color_type_);
    size_t n_total = 3 * (resolution_ - 1) * (resolution_ - 1);
    pointcloud->points_.resize(n_total);
    pointcloud->normals_.resize(n_total);
    pointcloud->colors_.resize(n_total);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      make_tuple_iterator(pointcloud->points_.begin(),
                                          pointcloud->normals_.begin(),
                                          pointcloud->colors_.begin()),
                      func);
    pointcloud->RemoveNoneFinitePoints(true, false);
    return pointcloud;
}

std::shared_ptr<geometry::TriangleMesh>
UniformTSDFVolume::ExtractTriangleMesh() {
    // implementation of marching cubes, based on
    // http://paulbourke.net/geometry/polygonise/
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    size_t res3 = (resolution_ - 1) * (resolution_ - 1) * (resolution_ - 1);
    size_t n_total = 8 * res3;

    // compute cube indices for each voxels
    thrust::device_vector<float> fs(n_total);
    thrust::device_vector<Eigen::Vector3f> cs(n_total);
    thrust::device_vector<Eigen::Vector3i> keys(res3);
    thrust::device_vector<Eigen::Vector3i> repeat_keys(n_total);
    thrust::device_vector<int> cube_indices(n_total);
    thrust::device_vector<int> cube_indices_out(res3);
    extract_mesh_phase1_functor func1(thrust::raw_pointer_cast(voxels_.data()),
                                      resolution_, color_type_);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      make_tuple_iterator(repeat_keys.begin(), cube_indices.begin(),
                                          fs.begin(), cs.begin()),
                      func1);
    thrust::reduce_by_key(repeat_keys.begin(), repeat_keys.end(),
                          cube_indices.begin(), keys.begin(),
                          cube_indices_out.begin(),
                          thrust::equal_to<Eigen::Vector3i>(),
                          add_cube_index_functor());
    auto begin1 = make_tuple_iterator(keys.begin(), cube_indices_out.begin());
    auto end1 = thrust::remove_if(begin1,
                                  make_tuple_iterator(keys.end(), cube_indices_out.end()),
                                  [] __device__ (const thrust::tuple<Eigen::Vector3i, int>& x) -> bool {
                                      int cidx = thrust::get<1>(x);
                                      return (cidx <= 0 || cidx == 255) ? true : false;
                                  });
    size_t n_result1 = thrust::distance(begin1, end1);
    keys.resize(n_result1);
    cube_indices_out.resize(n_result1);

    // compute vertices and vertex_colors
    thrust::repeated_range<thrust::device_vector<Eigen::Vector3i>::iterator> range_keys(keys.begin(),
                                                                                        keys.end(), 12);
    thrust::repeated_range<thrust::device_vector<int>::iterator> range_cube_indices(cube_indices_out.begin(),
                                                                                    cube_indices_out.end(), 12);
    size_t n_result2 = 12 * keys.size();
    mesh->vertices_.resize(n_result2);
    mesh->vertex_colors_.resize(n_result2);
    thrust::device_vector<Eigen::Vector4i> edge_indices(n_result2);
    cube_indices.resize(n_result2);
    repeat_keys.resize(n_result2);
    thrust::device_vector<int> vert_no(n_result2);
    extract_mesh_phase2_functor func2(origin_, voxel_length_,
                                      resolution_,
                                      thrust::raw_pointer_cast(fs.data()),
                                      thrust::raw_pointer_cast(cs.data()),
                                      color_type_);
    thrust::transform(make_tuple_iterator(range_keys.begin(), range_cube_indices.begin()),
                      make_tuple_iterator(range_keys.end(), range_cube_indices.end()),
                      thrust::make_counting_iterator<size_t>(0),
                      make_tuple_iterator(edge_indices.begin(), repeat_keys.begin(),
                                          cube_indices.begin(), vert_no.begin(),
                                          mesh->vertices_.begin(), mesh->vertex_colors_.begin()),
                      func2);
    auto begin2 = make_tuple_iterator(edge_indices.begin(), repeat_keys.begin(), cube_indices.begin(),
                                      vert_no.begin(), mesh->vertices_.begin(),
                                      mesh->vertex_colors_.begin());
    auto end2 = thrust::remove_if(begin2,
                                  make_tuple_iterator(edge_indices.end(), repeat_keys.end(),
                                                      cube_indices.end(),
                                                      vert_no.end(), mesh->vertices_.end(),
                                                      mesh->vertex_colors_.end()),
                                  [] __device__ (const thrust::tuple<Eigen::Vector4i, Eigen::Vector3i, int, int, Eigen::Vector3f, Eigen::Vector3f>& x) {
                                      Eigen::Vector4i edge_index = thrust::get<0>(x);
                                      return edge_index[0] < 0;
                                  });
    size_t n_result3 = thrust::distance(begin2, end2);
    edge_indices.resize(n_result3);
    repeat_keys.resize(n_result3);
    cube_indices.resize(n_result3);
    vert_no.resize(n_result3);
    mesh->vertices_.resize(n_result3);
    mesh->vertex_colors_.resize(n_result3);
    thrust::sort_by_key(edge_indices.begin(), edge_indices.end(),
                        make_tuple_iterator(cube_indices.begin(),
                                            repeat_keys.begin(),
                                            vert_no.begin(),
                                            mesh->vertices_.begin(),
                                            mesh->vertex_colors_.begin()));
    auto end3 = thrust::unique_by_key(edge_indices.begin(), edge_indices.end(),
                                      make_tuple_iterator(cube_indices.begin(),
                                                          repeat_keys.begin(),
                                                          vert_no.begin(),
                                                          mesh->vertices_.begin(),
                                                          mesh->vertex_colors_.begin()));
    size_t n_result4 = thrust::distance(edge_indices.begin(), end3.first);
    edge_indices.resize(n_result4);
    repeat_keys.resize(n_result4);
    cube_indices.resize(n_result4);
    vert_no.resize(n_result4);
    mesh->vertices_.resize(n_result4);
    mesh->vertex_colors_.resize(n_result4);
    thrust::sort_by_key(repeat_keys.begin(), repeat_keys.end(),
                        make_tuple_iterator(cube_indices.begin(),
                                            vert_no.begin(),
                                            mesh->vertices_.begin(),
                                            mesh->vertex_colors_.begin()));

    // compute triangles
    thrust::device_vector<int> seq(n_result4);
    thrust::sequence(seq.begin(), seq.end(), 0);
    auto end4 = thrust::unique_by_key(repeat_keys.begin(), repeat_keys.end(),
                                      seq.begin());
    size_t n_result5 = thrust::distance(repeat_keys.begin(), end4.first);
    repeat_keys.resize(n_result5);
    seq.resize(n_result5);
    seq.push_back(n_result4);
    mesh->triangles_.resize(n_result5 * 5);
    thrust::fill(mesh->triangles_.begin(), mesh->triangles_.end(), Eigen::Vector3i(-1, -1, -1));
    extract_mesh_phase3_functor func3(thrust::raw_pointer_cast(cube_indices.data()),
                                      thrust::raw_pointer_cast(vert_no.data()),
                                      thrust::raw_pointer_cast(seq.data()),
                                      thrust::raw_pointer_cast(mesh->triangles_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(n_result5), func3);
    auto end5 = thrust::remove_if(mesh->triangles_.begin(), mesh->triangles_.end(),
                                  [] __device__ (const Eigen::Vector3i& idxs) {return idxs[0] < 0;});
    mesh->triangles_.resize(thrust::distance(mesh->triangles_.begin(), end5));
    return mesh;
}

std::shared_ptr<geometry::PointCloud>
UniformTSDFVolume::ExtractVoxelPointCloud() const {
    auto voxel = std::make_shared<geometry::PointCloud>();
    // const float *p_tsdf = (const float *)tsdf_.data();
    // const float *p_weight = (const float *)weight_.data();
    // const float *p_color = (const float *)color_.data();
    extract_voxel_pointcloud_functor func(thrust::raw_pointer_cast(voxels_.data()),
                                          origin_, resolution_, voxel_length_);
    size_t n_total = resolution_ * resolution_ * resolution_;
    voxel->points_.resize(n_total);
    voxel->colors_.resize(n_total);
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(n_total),
                      make_tuple_iterator(voxel->points_.begin(), voxel->colors_.begin()), func);
    voxel->RemoveNoneFinitePoints(true, false);
    return voxel;
}

std::shared_ptr<geometry::VoxelGrid> UniformTSDFVolume::ExtractVoxelGrid()
        const {
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid->voxel_size_ = voxel_length_;
    voxel_grid->origin_ = origin_;
    size_t n_total = resolution_ * resolution_ * resolution_;
    voxel_grid->voxels_keys_.resize(n_total);
    voxel_grid->voxels_values_.resize(n_total);
    extract_voxel_grid_functor func(thrust::raw_pointer_cast(voxels_.data()), resolution_);
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(n_total),
                      make_tuple_iterator(voxel_grid->voxels_keys_.begin(), voxel_grid->voxels_values_.begin()),
                      func);
    auto begin = make_tuple_iterator(voxel_grid->voxels_keys_.begin(), voxel_grid->voxels_values_.begin());
    auto end = thrust::remove_if(begin,
                                 make_tuple_iterator(voxel_grid->voxels_keys_.end(), voxel_grid->voxels_values_.end()),
                                 [] __device__ (const thrust::tuple<Eigen::Vector3i, geometry::Voxel>& x) -> bool {
                                     Eigen::Vector3i index = thrust::get<0>(x);
                                     return index[0] < 0;
                                 });
    size_t n_out = thrust::distance(begin, end);
    voxel_grid->voxels_keys_.resize(n_out);
    voxel_grid->voxels_values_.resize(n_out);
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
    integrate_functor func(origin_, fx, fy, cx, cy, extrinsic, voxel_length_, sdf_trunc_,
                           safe_width, safe_height, resolution_,
                           thrust::raw_pointer_cast(image.color_.data_.data()),
                           thrust::raw_pointer_cast(image.depth_.data_.data()),
                           thrust::raw_pointer_cast(depth_to_camera_distance_multiplier.data_.data()),
                           image.depth_.width_, image.color_.num_of_channels_, color_type_,
                           thrust::raw_pointer_cast(voxels_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(resolution_ * resolution_ * resolution_),
                     func);
}