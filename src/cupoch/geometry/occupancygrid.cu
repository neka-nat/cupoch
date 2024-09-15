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
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/set_operations.h>
#include <thrust/iterator/discard_iterator.h>

#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/densegrid.inl"
#include "cupoch/geometry/geometry_functor.h"
#include "cupoch/geometry/geometry_utils.h"
#include "cupoch/geometry/intersection_test.h"
#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/utility/eigen.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace geometry {

namespace {

struct extract_range_voxels_functor {
    extract_range_voxels_functor(const Eigen::Vector3i& extents,
                                 int resolution,
                                 const Eigen::Vector3i& min_bound)
        : extents_(extents), resolution_(resolution), min_bound_(min_bound){};
    const Eigen::Vector3i extents_;
    const int resolution_;
    const Eigen::Vector3i min_bound_;
    __device__ int operator()(size_t idx) const {
        int x = idx / (extents_[1] * extents_[2]);
        int yz = idx % (extents_[1] * extents_[2]);
        int y = yz / extents_[2];
        int z = yz % extents_[2];
        Eigen::Vector3i gidx = min_bound_ + Eigen::Vector3i(x, y, z);
        return IndexOf(gidx, resolution_);
    }
};

__device__ int VoxelTraversal(Eigen::Vector3i* voxels,
                              int n_buffer,
                              const Eigen::Vector3i& half_resolution,
                              const Eigen::Vector3f& start,
                              const Eigen::Vector3f& end,
                              float voxel_size) {
    int n_voxels = 0;
    Eigen::Vector3f ray = end - start;
    float length = ray.norm();
    if (length == 0) {
        return n_voxels;
    }
    ray /= length;

    Eigen::Vector3i current_voxel(floorf(start[0] / voxel_size),
                                  floorf(start[1] / voxel_size),
                                  floorf(start[2] / voxel_size));
    Eigen::Vector3i last_voxel(floorf(end[0] / voxel_size),
                               floorf(end[1] / voxel_size),
                               floorf(end[2] / voxel_size));
    float stepX = (ray[0] > 0) ? 1 : ((ray[0] < 0) ? -1 : 0);
    float stepY = (ray[1] > 0) ? 1 : ((ray[1] < 0) ? -1 : 0);
    float stepZ = (ray[2] > 0) ? 1 : ((ray[2] < 0) ? -1 : 0);
    float voxel_boundary_x = (current_voxel[0] + 0.5 * stepX) * voxel_size;
    float voxel_boundary_y = (current_voxel[1] + 0.5 * stepY) * voxel_size;
    float voxel_boundary_z = (current_voxel[2] + 0.5 * stepZ) * voxel_size;
    float tMaxX = (stepX != 0) ? (voxel_boundary_x - start[0]) / ray[0]
                               : std::numeric_limits<float>::infinity();
    float tMaxY = (stepY != 0) ? (voxel_boundary_y - start[1]) / ray[1]
                               : std::numeric_limits<float>::infinity();
    float tMaxZ = (stepZ != 0) ? (voxel_boundary_z - start[2]) / ray[2]
                               : std::numeric_limits<float>::infinity();
    float tDeltaX = (stepX != 0) ? voxel_size / fabs(ray[0])
                                 : std::numeric_limits<float>::infinity();
    float tDeltaY = (stepY != 0) ? voxel_size / fabs(ray[1])
                                 : std::numeric_limits<float>::infinity();
    float tDeltaZ = (stepZ != 0) ? voxel_size / fabs(ray[2])
                                 : std::numeric_limits<float>::infinity();

    voxels[n_voxels] = current_voxel + half_resolution;
    ++n_voxels;

    while (n_voxels < n_buffer) {
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                current_voxel[0] += stepX;
                tMaxX += tDeltaX;
            } else {
                current_voxel[2] += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                current_voxel[1] += stepY;
                tMaxY += tDeltaY;
            } else {
                current_voxel[2] += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
        if (last_voxel == current_voxel) {
            break;
        } else {
            float dist_from_origin = min(min(tMaxX, tMaxY), tMaxZ);
            if (dist_from_origin > length) {
                break;
            } else {
                voxels[n_voxels] = current_voxel + half_resolution;
                ++n_voxels;
            }
        }
    }
    return n_voxels;
}

struct compute_voxel_traversal_functor {
    compute_voxel_traversal_functor(Eigen::Vector3i* voxels,
                                    int n_step,
                                    const Eigen::Vector3f& viewpoint,
                                    const Eigen::Vector3i& half_resolution,
                                    float voxel_size,
                                    const Eigen::Vector3f& origin)
        : voxels_(voxels),
          n_step_(n_step),
          viewpoint_(viewpoint),
          half_resolution_(half_resolution),
          voxel_size_(voxel_size),
          origin_(origin){};
    Eigen::Vector3i* voxels_;
    const int n_step_;
    const Eigen::Vector3f viewpoint_;
    const Eigen::Vector3i half_resolution_;
    const float voxel_size_;
    const Eigen::Vector3f origin_;
    __device__ void operator()(
            const thrust::tuple<size_t, Eigen::Vector3f>& x) {
        const int idx = thrust::get<0>(x);
        const Eigen::Vector3f end = thrust::get<1>(x);
        VoxelTraversal(voxels_ + idx * n_step_, n_step_, half_resolution_,
                       viewpoint_, end - origin_, voxel_size_);
    }
};

void ComputeFreeVoxels(const utility::device_vector<Eigen::Vector3f>& points,
                       const Eigen::Vector3f& viewpoint,
                       float voxel_size,
                       int resolution,
                       Eigen::Vector3f& origin,
                       int n_div,
                       utility::device_vector<Eigen::Vector3i>& free_voxels) {
    if (points.empty()) return;
    size_t n_points = points.size();
    Eigen::Vector3i half_resolution = Eigen::Vector3i::Constant(resolution / 2);
    free_voxels.resize(
            n_div * 3 * n_points,
            Eigen::Vector3i::Constant(geometry::INVALID_VOXEL_INDEX));
    compute_voxel_traversal_functor func(
            thrust::raw_pointer_cast(free_voxels.data()), n_div * 3,
            viewpoint - origin, half_resolution, voxel_size, origin);
    thrust::for_each(enumerate_begin(points), enumerate_end(points), func);
    auto end1 = thrust::remove_if(
            free_voxels.begin(), free_voxels.end(),
            [resolution] __device__(const Eigen::Vector3i& idx) -> bool {
                return idx[0] < 0 || idx[1] < 0 || idx[2] < 0 ||
                       idx[0] >= resolution || idx[1] >= resolution ||
                       idx[2] >= resolution;
            });
    free_voxels.resize(thrust::distance(free_voxels.begin(), end1));
    thrust::sort(utility::exec_policy(0), free_voxels.begin(),
                 free_voxels.end());
    auto end2 = thrust::unique(utility::exec_policy(0),
                               free_voxels.begin(), free_voxels.end());
    free_voxels.resize(thrust::distance(free_voxels.begin(), end2));
}

struct create_occupancy_voxels_functor {
    create_occupancy_voxels_functor(const Eigen::Vector3f& origin,
                                    const Eigen::Vector3i& half_resolution,
                                    float voxel_size)
        : origin_(origin),
          half_resolution_(half_resolution),
          voxel_size_(voxel_size){};
    const Eigen::Vector3f origin_;
    const Eigen::Vector3i half_resolution_;
    const float voxel_size_;
    __device__ Eigen::Vector3i operator()(
            const thrust::tuple<Eigen::Vector3f, bool>& x) const {
        const Eigen::Vector3f& point = thrust::get<0>(x);
        bool hit_flag = thrust::get<1>(x);
        Eigen::Vector3f ref_coord = (point - origin_) / voxel_size_;
        return (hit_flag)
                       ? Eigen::device_vectorize<float, 3, ::floor>(ref_coord)
                                         .cast<int>() +
                                 half_resolution_
                       : Eigen::Vector3i(INVALID_VOXEL_INDEX,
                                         INVALID_VOXEL_INDEX,
                                         INVALID_VOXEL_INDEX);
    }
};

void ComputeOccupiedVoxels(
        const utility::device_vector<Eigen::Vector3f>& points,
        const utility::device_vector<bool> hit_flags,
        float voxel_size,
        int resolution,
        Eigen::Vector3f& origin,
        utility::device_vector<Eigen::Vector3i>& occupied_voxels) {
    occupied_voxels.resize(points.size());
    Eigen::Vector3i half_resolution = Eigen::Vector3i::Constant(resolution / 2);
    create_occupancy_voxels_functor func(origin, half_resolution, voxel_size);
    thrust::transform(make_tuple_begin(points, hit_flags),
                      make_tuple_end(points, hit_flags),
                      occupied_voxels.begin(), func);
    auto end1 = thrust::remove_if(
            occupied_voxels.begin(), occupied_voxels.end(),
            [resolution] __device__(const Eigen::Vector3i& idx) -> bool {
                return idx[0] < 0 || idx[1] < 0 || idx[2] < 0 ||
                       idx[0] >= resolution || idx[1] >= resolution ||
                       idx[2] >= resolution;
            });
    occupied_voxels.resize(thrust::distance(occupied_voxels.begin(), end1));
    thrust::sort(utility::exec_policy(0), occupied_voxels.begin(),
                 occupied_voxels.end());
    auto end2 = thrust::unique(utility::exec_policy(0),
                               occupied_voxels.begin(), occupied_voxels.end());
    occupied_voxels.resize(thrust::distance(occupied_voxels.begin(), end2));
}

struct add_occupancy_functor {
    add_occupancy_functor(OccupancyVoxel* voxels,
                          int resolution,
                          float clamping_thres_min,
                          float clamping_thres_max,
                          float prob_miss_log,
                          float prob_hit_log,
                          bool occupied)
        : voxels_(voxels),
          resolution_(resolution),
          clamping_thres_min_(clamping_thres_min),
          clamping_thres_max_(clamping_thres_max),
          prob_miss_log_(prob_miss_log),
          prob_hit_log_(prob_hit_log),
          occupied_(occupied){};
    OccupancyVoxel* voxels_;
    const int resolution_;
    const float clamping_thres_min_;
    const float clamping_thres_max_;
    const float prob_miss_log_;
    const float prob_hit_log_;
    const bool occupied_;
    __device__ void operator()(const Eigen::Vector3i& voxel) {
        size_t idx = IndexOf(voxel, resolution_);
        float p = voxels_[idx].prob_log_;
        p = (isnan(p)) ? 0 : p;
        p += (occupied_) ? prob_hit_log_ : prob_miss_log_;
        voxels_[idx].prob_log_ =
                min(max(p, clamping_thres_min_), clamping_thres_max_);
        voxels_[idx].grid_index_ = voxel.cast<unsigned short>();
    }
};

}  // namespace

template class DenseGrid<OccupancyVoxel>;

OccupancyGrid::OccupancyGrid()
    : DenseGrid<OccupancyVoxel>(Geometry::GeometryType::OccupancyGrid,
                                0.05,
                                512,
                                Eigen::Vector3f::Zero()),
      min_bound_(Eigen::Vector3ui16::Constant(resolution_ / 2)),
      max_bound_(Eigen::Vector3ui16::Constant(resolution_ / 2)) {}
OccupancyGrid::OccupancyGrid(float voxel_size,
                             size_t resolution,
                             const Eigen::Vector3f& origin)
    : DenseGrid<OccupancyVoxel>(Geometry::GeometryType::OccupancyGrid,
                                voxel_size,
                                resolution,
                                origin),
      min_bound_(Eigen::Vector3ui16::Constant(resolution_ / 2)),
      max_bound_(Eigen::Vector3ui16::Constant(resolution_ / 2)) {}
OccupancyGrid::~OccupancyGrid() {}
OccupancyGrid::OccupancyGrid(const OccupancyGrid& other)
    : DenseGrid<OccupancyVoxel>(Geometry::GeometryType::OccupancyGrid, other),
      min_bound_(other.min_bound_),
      max_bound_(other.max_bound_),
      clamping_thres_min_(other.clamping_thres_min_),
      clamping_thres_max_(other.clamping_thres_max_),
      prob_hit_log_(other.prob_hit_log_),
      prob_miss_log_(other.prob_miss_log_),
      occ_prob_thres_log_(other.occ_prob_thres_log_),
      visualize_free_area_(other.visualize_free_area_) {}

OccupancyGrid& OccupancyGrid::Clear() {
    DenseGrid::Clear();
    min_bound_ = Eigen::Vector3ui16::Constant(resolution_ / 2);
    max_bound_ = Eigen::Vector3ui16::Constant(resolution_ / 2);
    return *this;
}

Eigen::Vector3f OccupancyGrid::GetMinBound() const {
    return (min_bound_.cast<int>() - Eigen::Vector3i::Constant(resolution_ / 2))
                           .cast<float>() *
                   voxel_size_ +
           origin_;
}

Eigen::Vector3f OccupancyGrid::GetMaxBound() const {
    return (max_bound_.cast<int>() -
            Eigen::Vector3i::Constant(resolution_ / 2 - 1))
                           .cast<float>() *
                   voxel_size_ +
           origin_;
}

bool OccupancyGrid::IsOccupied(const Eigen::Vector3f& point) const {
    auto idx = GetVoxelIndex(point);
    if (idx < 0) return false;
    OccupancyVoxel voxel = voxels_[idx];
    return !std::isnan(voxel.prob_log_) &&
           voxel.prob_log_ > occ_prob_thres_log_;
}

bool OccupancyGrid::IsUnknown(const Eigen::Vector3f& point) const {
    auto idx = GetVoxelIndex(point);
    if (idx < 0) return true;
    OccupancyVoxel voxel = voxels_[idx];
    return std::isnan(voxel.prob_log_);
}

thrust::tuple<bool, OccupancyVoxel> OccupancyGrid::GetVoxel(
        const Eigen::Vector3f& point) const {
    auto idx = GetVoxelIndex(point);
    if (idx < 0) return thrust::make_tuple(false, OccupancyVoxel());
    OccupancyVoxel voxel = voxels_[idx];
    return thrust::make_tuple(!std::isnan(voxel.prob_log_), voxel);
}

template <typename Func>
std::shared_ptr<utility::device_vector<OccupancyVoxel>>
OccupancyGrid::ExtractBoundVoxels(Func check_func) const {
    Eigen::Vector3ui16 diff =
            max_bound_ - min_bound_ + Eigen::Vector3ui16::Ones();
    auto out = std::make_shared<utility::device_vector<OccupancyVoxel>>();
    out->resize(diff[0] * diff[1] * diff[2]);
    extract_range_voxels_functor func(diff.cast<int>(), resolution_,
                                      min_bound_.cast<int>());
    auto end = thrust::copy_if(
            thrust::make_permutation_iterator(
                    voxels_.begin(),
                    thrust::make_transform_iterator(
                            thrust::make_counting_iterator<size_t>(0), func)),
            thrust::make_permutation_iterator(
                    voxels_.begin(),
                    thrust::make_transform_iterator(
                            thrust::make_counting_iterator(out->size()), func)),
            out->begin(), check_func);
    out->resize(thrust::distance(out->begin(), end));
    return out;
}

std::shared_ptr<utility::device_vector<OccupancyVoxel>>
OccupancyGrid::ExtractKnownVoxels() const {
    auto check_fn = [th = occ_prob_thres_log_] __device__(
                            const thrust::tuple<OccupancyVoxel>& x) {
        const OccupancyVoxel& v = thrust::get<0>(x);
        return !isnan(v.prob_log_);
    };
    return ExtractBoundVoxels(check_fn);
}

std::shared_ptr<utility::device_vector<OccupancyVoxel>>
OccupancyGrid::ExtractFreeVoxels() const {
    auto check_fn = [th = occ_prob_thres_log_] __device__(
                            const thrust::tuple<OccupancyVoxel>& x) {
        const OccupancyVoxel& v = thrust::get<0>(x);
        return !isnan(v.prob_log_) && v.prob_log_ <= th;
    };
    return ExtractBoundVoxels(check_fn);
}

std::shared_ptr<utility::device_vector<OccupancyVoxel>>
OccupancyGrid::ExtractOccupiedVoxels() const {
    auto check_fn = [th = occ_prob_thres_log_] __device__(
                            const thrust::tuple<OccupancyVoxel>& x) {
        const OccupancyVoxel& v = thrust::get<0>(x);
        return !isnan(v.prob_log_) && v.prob_log_ > th;
    };
    return ExtractBoundVoxels(check_fn);
}

OccupancyGrid& OccupancyGrid::Reconstruct(float voxel_size, int resolution) {
    DenseGrid::Reconstruct(voxel_size, resolution);
    return *this;
}

OccupancyGrid& OccupancyGrid::SetFreeArea(const Eigen::Vector3f& min_bound,
                                          const Eigen::Vector3f& max_bound) {
    const Eigen::Vector3i half_res = Eigen::Vector3i::Constant(resolution_ / 2);
    Eigen::Vector3i imin_bound = ((min_bound - origin_) / voxel_size_)
                                         .array()
                                         .floor()
                                         .matrix()
                                         .cast<int>() +
                                 half_res;
    Eigen::Vector3i imax_bound = ((max_bound - origin_) / voxel_size_)
                                         .array()
                                         .floor()
                                         .matrix()
                                         .cast<int>() +
                                 half_res;
    min_bound_ = imin_bound.array()
                         .max(Eigen::Array3i(0, 0, 0))
                         .matrix()
                         .cast<unsigned short>();
    max_bound_ = imax_bound.array()
                         .min(Eigen::Array3i(resolution_ - 1, resolution_ - 1,
                                             resolution_ - 1))
                         .matrix()
                         .cast<unsigned short>();
    Eigen::Vector3ui16 diff =
            max_bound_ - min_bound_ + Eigen::Vector3ui16::Ones();
    extract_range_voxels_functor func(diff.cast<int>(), resolution_,
                                      min_bound_.cast<int>());
    thrust::for_each(
            thrust::make_permutation_iterator(
                    voxels_.begin(),
                    thrust::make_transform_iterator(
                            thrust::make_counting_iterator<size_t>(0), func)),
            thrust::make_permutation_iterator(
                    voxels_.begin(),
                    thrust::make_transform_iterator(
                            thrust::make_counting_iterator<size_t>(
                                    diff[0] * diff[1] * diff[2]),
                            func)),
            [pml = prob_miss_log_] __device__(geometry::OccupancyVoxel & v) {
                v.prob_log_ = (isnan(v.prob_log_)) ? 0 : v.prob_log_;
                v.prob_log_ += pml;
            });
    return *this;
}

OccupancyGrid& OccupancyGrid::Insert(
        const utility::device_vector<Eigen::Vector3f>& points,
        const Eigen::Vector3f& viewpoint,
        float max_range) {
    if (points.empty()) return *this;

    utility::device_vector<Eigen::Vector3f> ranged_points(points.size());
    utility::device_vector<float> ranged_dists(points.size());
    utility::device_vector<bool> hit_flags(points.size());

    thrust::transform(
            points.begin(), points.end(),
            make_tuple_begin(ranged_points, ranged_dists, hit_flags),
            [viewpoint, max_range] __device__(const Eigen::Vector3f& pt) {
                const Eigen::Vector3f pt_vp = pt - viewpoint;
                const float dist = pt_vp.norm();
                const bool is_hit = max_range < 0 || dist <= max_range;
                const Eigen::Vector3f ranged_pt =
                        (is_hit)
                                ? pt
                                : ((dist == 0) ? viewpoint
                                               : viewpoint + pt_vp / dist *
                                                                     max_range);
                return thrust::make_tuple(
                        ranged_pt,
                        (ranged_pt - viewpoint).array().abs().maxCoeff(),
                        is_hit);
            });
    float max_dist =
            *(thrust::max_element(ranged_dists.begin(), ranged_dists.end()));
    int n_div = int(std::ceil(max_dist / voxel_size_));

    utility::device_vector<Eigen::Vector3i> free_voxels;
    utility::device_vector<Eigen::Vector3i> occupied_voxels;
    if (n_div > 0) {
        // comupute free voxels
        ComputeFreeVoxels(ranged_points, viewpoint, voxel_size_, resolution_,
                          origin_, n_div + 1, free_voxels);
    }
    // compute occupied voxels
    ComputeOccupiedVoxels(ranged_points, hit_flags, voxel_size_, resolution_,
                          origin_, occupied_voxels);

    if (n_div > 0) {
        utility::device_vector<Eigen::Vector3i> free_voxels_res(
                free_voxels.size());
        auto end = thrust::set_difference(
                free_voxels.begin(), free_voxels.end(), occupied_voxels.begin(),
                occupied_voxels.end(), free_voxels_res.begin());
        free_voxels_res.resize(thrust::distance(free_voxels_res.begin(), end));
        AddVoxels(free_voxels_res, false);
    }
    AddVoxels(occupied_voxels, true);
    return *this;
}

OccupancyGrid& OccupancyGrid::Insert(
        const utility::pinned_host_vector<Eigen::Vector3f>& points,
        const Eigen::Vector3f& viewpoint,
        float max_range) {
    utility::device_vector<Eigen::Vector3f> dev_points(points.size());
    cudaSafeCall(cudaMemcpy(
            thrust::raw_pointer_cast(dev_points.data()), thrust::raw_pointer_cast(points.data()),
            points.size() * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    return Insert(dev_points, viewpoint, max_range);
}

OccupancyGrid& OccupancyGrid::Insert(
        const thrust::host_vector<Eigen::Vector3f>& points,
        const Eigen::Vector3f& viewpoint,
        float max_range) {
    utility::device_vector<Eigen::Vector3f> dev_points(points.size());
    cudaSafeCall(cudaMemcpy(
            thrust::raw_pointer_cast(dev_points.data()), thrust::raw_pointer_cast(points.data()),
            points.size() * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    return Insert(dev_points, viewpoint, max_range);
}

OccupancyGrid& OccupancyGrid::Insert(const std::vector<Eigen::Vector3f>& points,
                                     const Eigen::Vector3f& viewpoint,
                                     float max_range) {
    utility::device_vector<Eigen::Vector3f> dev_points(points.size());
    copy_host_to_device(points, dev_points);
    return Insert(dev_points, viewpoint, max_range);
}

OccupancyGrid& OccupancyGrid::Insert(const geometry::PointCloud& pointcloud,
                                     const Eigen::Vector3f& viewpoint,
                                     float max_range) {
    Insert(pointcloud.points_, viewpoint, max_range);
    return *this;
}

OccupancyGrid& OccupancyGrid::AddVoxel(const Eigen::Vector3i& voxel,
                                       bool occupied) {
    int idx = IndexOf(voxel, resolution_);
    size_t max_idx = resolution_ * resolution_ * resolution_;
    if (idx < 0 || idx >= max_idx) {
        utility::LogError(
                "[OccupancyGrid] a provided voxeld is not occupancy grid "
                "range.");
        return *this;
    } else {
        OccupancyVoxel org_ov = voxels_[idx];
        if (std::isnan(org_ov.prob_log_)) org_ov.prob_log_ = 0.0;
        org_ov.prob_log_ += (occupied) ? prob_hit_log_ : prob_miss_log_;
        org_ov.prob_log_ =
                std::min(std::max(org_ov.prob_log_, clamping_thres_min_),
                         clamping_thres_max_);
        org_ov.grid_index_ = voxel.cast<unsigned short>();
        voxels_[idx] = org_ov;
        min_bound_ = min_bound_.array().min(org_ov.grid_index_.array());
        max_bound_ = max_bound_.array().max(org_ov.grid_index_.array());
    }
    return *this;
}

OccupancyGrid& OccupancyGrid::AddVoxels(
        const utility::device_vector<Eigen::Vector3i>& voxels, bool occupied) {
    if (voxels.empty()) return *this;
    Eigen::Vector3i minv = utility::ComputeMinBound<3, int>(voxels);
    Eigen::Vector3i maxv = utility::ComputeMaxBound<3, int>(voxels);
    Eigen::Vector3ui16 minvu = minv.cast<unsigned short>();
    Eigen::Vector3ui16 maxvu = maxv.cast<unsigned short>();
    min_bound_ = min_bound_.array().min(minvu.array());
    min_bound_ = min_bound_.array().min(maxvu.array());
    max_bound_ = max_bound_.array().max(minvu.array());
    max_bound_ = max_bound_.array().max(maxvu.array());
    add_occupancy_functor func(thrust::raw_pointer_cast(voxels_.data()),
                               resolution_, clamping_thres_min_,
                               clamping_thres_max_, prob_miss_log_,
                               prob_hit_log_, occupied);
    thrust::for_each(voxels.begin(), voxels.end(), func);
    return *this;
}

}  // namespace geometry
}  // namespace cupoch