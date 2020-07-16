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

#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/densegrid.inl"
#include "cupoch/geometry/geometry_functor.h"
#include "cupoch/geometry/intersection_test.h"
#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace geometry {

namespace {

__constant__ float voxel_offset[7][3] = {{0, 0, 0}, {1, 0, 0},  {-1, 0, 0},
                                         {0, 1, 0}, {0, -1, 0}, {0, 0, 1},
                                         {0, 0, -1}};

struct extract_range_voxels_functor {
    extract_range_voxels_functor(const OccupancyVoxel* voxels,
                                 const Eigen::Vector3i& extents,
                                 int resolution,
                                 const Eigen::Vector3i& min_bound)
        : voxels_(voxels),
          extents_(extents),
          resolution_(resolution),
          min_bound_(min_bound){};
    const OccupancyVoxel* voxels_;
    const Eigen::Vector3i extents_;
    const int resolution_;
    const Eigen::Vector3i min_bound_;
    __device__ OccupancyVoxel operator()(size_t idx) const {
        int x = idx / (extents_[1] * extents_[2]);
        int yz = idx % (extents_[1] * extents_[2]);
        int y = yz / extents_[2];
        int z = yz % extents_[2];
        Eigen::Vector3i gidx = min_bound_ + Eigen::Vector3i(x, y, z);
        return voxels_[IndexOf(gidx, resolution_)];
    }
};

struct compute_intersect_voxel_segment_functor {
    compute_intersect_voxel_segment_functor(
            const Eigen::Vector3f* points,
            const Eigen::Vector3f* steps,
            const Eigen::Vector3f& viewpoint,
            const Eigen::Vector3i& half_resolution,
            float voxel_size,
            const Eigen::Vector3f& origin,
            int n_div)
        : points_(points),
          steps_(steps),
          viewpoint_(viewpoint),
          half_resolution_(half_resolution),
          voxel_size_(voxel_size),
          box_half_size_(Eigen::Vector3f(
                  voxel_size / 2, voxel_size / 2, voxel_size / 2)),
          origin_(origin),
          n_div_(n_div){};
    const Eigen::Vector3f* points_;
    const Eigen::Vector3f* steps_;
    const Eigen::Vector3f viewpoint_;
    const Eigen::Vector3i half_resolution_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const Eigen::Vector3f origin_;
    const int n_div_;
    __device__ Eigen::Vector3i operator()(size_t idx) {
        int pidx = idx / (n_div_ * 7);
        int svidx = idx % (n_div_ * 7);
        int sidx = svidx / 7;
        int vidx = svidx % 7;
        Eigen::Vector3f center = sidx * steps_[pidx] + viewpoint_;
        Eigen::Vector3f voxel_idx = Eigen::device_vectorize<float, 3, ::floor>(
                (center - origin_) / voxel_size_);
        Eigen::Vector3f voxel_center =
                voxel_size_ *
                (voxel_idx + Eigen::Vector3f(voxel_offset[vidx][0],
                                             voxel_offset[vidx][1],
                                             voxel_offset[vidx][2]));
        bool is_intersect = intersection_test::LineSegmentAABB(
                viewpoint_, points_[pidx], voxel_center - box_half_size_,
                voxel_center + box_half_size_);
        return (is_intersect) ? voxel_idx.cast<int>() + half_resolution_
                              : Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX,
                                                geometry::INVALID_VOXEL_INDEX,
                                                geometry::INVALID_VOXEL_INDEX);
    }
};

void ComputeFreeVoxels(const utility::device_vector<Eigen::Vector3f>& points,
                       const Eigen::Vector3f& viewpoint,
                       float voxel_size,
                       int resolution,
                       Eigen::Vector3f& origin,
                       const utility::device_vector<Eigen::Vector3f>& steps,
                       int n_div,
                       utility::device_vector<Eigen::Vector3i>& free_voxels) {
    if (points.empty()) return;
    size_t n_points = points.size();
    size_t max_idx = resolution * resolution * resolution;
    Eigen::Vector3i half_resolution = Eigen::Vector3i::Constant(resolution / 2);
    free_voxels.resize(n_div * n_points * 7);
    compute_intersect_voxel_segment_functor func(
            thrust::raw_pointer_cast(points.data()),
            thrust::raw_pointer_cast(steps.data()), viewpoint, half_resolution,
            voxel_size, origin, n_div);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_div * n_points * 7),
                      free_voxels.begin(), func);
    auto end1 = thrust::remove_if(
            free_voxels.begin(), free_voxels.end(),
            [max_idx] __device__(const Eigen::Vector3i& idx) -> bool {
                return idx[0] < 0 || idx[1] < 0 || idx[2] < 0 ||
                       idx[0] >= max_idx || idx[1] >= max_idx ||
                       idx[2] >= max_idx;
            });
    free_voxels.resize(thrust::distance(free_voxels.begin(), end1));
    thrust::sort(free_voxels.begin(), free_voxels.end());
    auto end2 = thrust::unique(free_voxels.begin(), free_voxels.end());
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
        ;
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
    size_t max_idx = resolution * resolution * resolution;
    Eigen::Vector3i half_resolution = Eigen::Vector3i::Constant(resolution / 2);
    create_occupancy_voxels_functor func(origin, half_resolution, voxel_size);
    thrust::transform(make_tuple_begin(points, hit_flags),
                      make_tuple_end(points, hit_flags),
                      occupied_voxels.begin(), func);
    auto end1 = thrust::remove_if(
            occupied_voxels.begin(), occupied_voxels.end(),
            [max_idx] __device__(const Eigen::Vector3i& idx) -> bool {
                return idx[0] < 0 || idx[1] < 0 || idx[2] < 0 ||
                       idx[0] >= max_idx || idx[1] >= max_idx ||
                       idx[2] >= max_idx;
            });
    occupied_voxels.resize(thrust::distance(occupied_voxels.begin(), end1));
    thrust::sort(occupied_voxels.begin(), occupied_voxels.end());
    auto end2 = thrust::unique(occupied_voxels.begin(), occupied_voxels.end());
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
                             int resolution,
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
                   voxel_size_ -
           origin_;
}

Eigen::Vector3f OccupancyGrid::GetMaxBound() const {
    return (max_bound_.cast<int>() -
            Eigen::Vector3i::Constant(resolution_ / 2 - 1))
                           .cast<float>() *
                   voxel_size_ -
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

std::shared_ptr<utility::device_vector<OccupancyVoxel>>
OccupancyGrid::ExtractBoundVoxels() const {
    Eigen::Vector3ui16 diff =
            max_bound_ - min_bound_ + Eigen::Vector3ui16::Ones();
    auto out = std::make_shared<utility::device_vector<OccupancyVoxel>>();
    out->resize(diff[0] * diff[1] * diff[2]);
    extract_range_voxels_functor func(thrust::raw_pointer_cast(voxels_.data()),
                                      diff.cast<int>(), resolution_,
                                      min_bound_.cast<int>());
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(out->size()), out->begin(),
                      func);
    return out;
}

std::shared_ptr<utility::device_vector<OccupancyVoxel>>
OccupancyGrid::ExtractKnownVoxels() const {
    auto out = ExtractBoundVoxels();
    auto remove_fn = [th = occ_prob_thres_log_] __device__(
                             const thrust::tuple<OccupancyVoxel>& x) {
        const OccupancyVoxel& v = thrust::get<0>(x);
        return isnan(v.prob_log_);
    };
    remove_if_vectors(remove_fn, *out);
    return out;
}

std::shared_ptr<utility::device_vector<OccupancyVoxel>>
OccupancyGrid::ExtractFreeVoxels() const {
    auto out = ExtractBoundVoxels();
    auto remove_fn = [th = occ_prob_thres_log_] __device__(
                             const thrust::tuple<OccupancyVoxel>& x) {
        const OccupancyVoxel& v = thrust::get<0>(x);
        return isnan(v.prob_log_) || v.prob_log_ > th;
    };
    remove_if_vectors(remove_fn, *out);
    return out;
}

std::shared_ptr<utility::device_vector<OccupancyVoxel>>
OccupancyGrid::ExtractOccupiedVoxels() const {
    auto out = ExtractBoundVoxels();
    auto remove_fn = [th = occ_prob_thres_log_] __device__(
                             const thrust::tuple<OccupancyVoxel>& x) {
        const OccupancyVoxel& v = thrust::get<0>(x);
        return isnan(v.prob_log_) || v.prob_log_ <= th;
    };
    remove_if_vectors(remove_fn, *out);
    return out;
}

OccupancyGrid& OccupancyGrid::Reconstruct(float voxel_size, int resolution) {
    DenseGrid::Reconstruct(voxel_size, resolution);
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
                Eigen::Vector3f pt_vp = pt - viewpoint;
                float dist = pt_vp.norm();
                bool is_hit = max_range < 0 || dist <= max_range;
                return thrust::make_tuple(
                        (is_hit) ? pt : viewpoint + pt_vp / dist * max_range,
                        (is_hit) ? dist : max_range, is_hit);
            });
    float max_dist =
            *(thrust::max_element(ranged_dists.begin(), ranged_dists.end()));
    int n_div = int(std::ceil(max_dist / voxel_size_));

    utility::device_vector<Eigen::Vector3i> free_voxels;
    utility::device_vector<Eigen::Vector3i> occupied_voxels;
    if (n_div > 0) {
        utility::device_vector<Eigen::Vector3f> steps(points.size());
        thrust::transform(
                ranged_points.begin(), ranged_points.end(), steps.begin(),
                [viewpoint, n_div] __device__(const Eigen::Vector3f& pt) {
                    return (pt - viewpoint) / n_div;
                });
        // comupute free voxels
        ComputeFreeVoxels(ranged_points, viewpoint, voxel_size_, resolution_,
                          origin_, steps, n_div + 1, free_voxels);
    } else {
        thrust::copy(points.begin(), points.end(), ranged_points.begin());
        thrust::fill(hit_flags.begin(), hit_flags.end(), true);
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
        const thrust::host_vector<Eigen::Vector3f>& points,
        const Eigen::Vector3f& viewpoint,
        float max_range) {
    utility::device_vector<Eigen::Vector3f> dev_points = points;
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
    Eigen::Vector3i fv = voxels.front();
    Eigen::Vector3i bv = voxels.back();
    Eigen::Vector3ui16 fvu = fv.cast<unsigned short>();
    Eigen::Vector3ui16 bvu = bv.cast<unsigned short>();
    min_bound_ = min_bound_.array().min(fvu.array());
    min_bound_ = min_bound_.array().min(bvu.array());
    max_bound_ = max_bound_.array().max(fvu.array());
    max_bound_ = max_bound_.array().max(bvu.array());
    add_occupancy_functor func(thrust::raw_pointer_cast(voxels_.data()),
                               resolution_, clamping_thres_min_,
                               clamping_thres_max_, prob_miss_log_,
                               prob_hit_log_, occupied);
    thrust::for_each(voxels.begin(), voxels.end(), func);
    return *this;
}

}  // namespace geometry
}  // namespace cupoch