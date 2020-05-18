#include "cupoch/collision/collision.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/geometry/lineset.h"
#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/geometry/intersection_test.h"

#include <thrust/iterator/discard_iterator.h>

namespace cupoch {
namespace collision {

namespace {

struct intersect_voxel_voxel_functor {
    intersect_voxel_voxel_functor(const Eigen::Vector3i* voxels_keys1,
                                  const Eigen::Vector3i* voxels_keys2,
                                  float voxel_size1, float voxel_size2,
                                  const Eigen::Vector3f& origin1,
                                  const Eigen::Vector3f& origin2,
                                  int n_v2, float margin)
                                  : voxels_keys1_(voxels_keys1), voxels_keys2_(voxels_keys2),
                                  voxel_size1_(voxel_size1), voxel_size2_(voxel_size2),
                                  box_half_size1_(Eigen::Vector3f(
                                    voxel_size1 / 2, voxel_size1 / 2, voxel_size1 / 2)),
                                  box_half_size2_(Eigen::Vector3f(
                                        voxel_size2 / 2, voxel_size2 / 2, voxel_size2 / 2)),
                                  origin1_(origin1), origin2_(origin2),
                                  n_v2_(n_v2), margin_(margin) {};
    const Eigen::Vector3i* voxels_keys1_;
    const Eigen::Vector3i* voxels_keys2_;
    const float voxel_size1_;
    const float voxel_size2_;
    const Eigen::Vector3f box_half_size1_;
    const Eigen::Vector3f box_half_size2_;
    const Eigen::Vector3f origin1_;
    const Eigen::Vector3f origin2_;
    const int n_v2_;
    const float margin_;
    __device__ Eigen::Vector2i operator() (size_t idx) const {
        int i1 = idx / n_v2_;
        int i2 = idx % n_v2_;
        const Eigen::Vector3f h3 = Eigen::Vector3f::Constant(0.5);
        const Eigen::Vector3f ms = Eigen::Vector3f::Constant(margin_);
        Eigen::Vector3f center1 = ((voxels_keys1_[i1].cast<float>() + h3) * voxel_size1_) + origin1_;
        Eigen::Vector3f center2 = ((voxels_keys2_[i2].cast<float>() + h3) * voxel_size2_) + origin2_;
        int coll = geometry::intersection_test::AABBAABB(center1 - box_half_size1_ - ms, center1 + box_half_size1_ + ms,
                                                         center2 - box_half_size2_, center2 + box_half_size2_);
        return (coll == 1) ? Eigen::Vector2i(i1, i2) : Eigen::Vector2i(-1, -1);
    }
};

struct intersect_voxel_line_functor {
    intersect_voxel_line_functor(const Eigen::Vector3i* voxels_keys,
                                 const Eigen::Vector3f* points,
                                 const Eigen::Vector2i* lines,
                                 float voxel_size,
                                 const Eigen::Vector3f& origin,
                                 int n_v2, float margin)
                                  : voxels_keys_(voxels_keys),
                                  points_(points), lines_(lines),
                                  voxel_size_(voxel_size),
                                  box_half_size_(Eigen::Vector3f(
                                    voxel_size / 2, voxel_size / 2, voxel_size / 2)),
                                  origin_(origin), n_v2_(n_v2), margin_(margin) {};
    const Eigen::Vector3i* voxels_keys_;
    const Eigen::Vector3f* points_;
    const Eigen::Vector2i* lines_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const Eigen::Vector3f origin_;
    const int n_v2_;
    const float margin_;
    __device__ Eigen::Vector2i operator() (size_t idx) const {
        int i1 = idx / n_v2_;
        int i2 = idx % n_v2_;
        const Eigen::Vector3f h3 = Eigen::Vector3f::Constant(0.5);
        const Eigen::Vector3f ms = Eigen::Vector3f::Constant(margin_);
        Eigen::Vector3f center = ((voxels_keys_[i1].cast<float>() + h3) * voxel_size_) + origin_;
        Eigen::Vector2i lidx = lines_[idx];
        int coll = geometry::intersection_test::LineSegmentAABB(points_[lidx[0]], points_[lidx[1]],
                                                                center - box_half_size_, center + box_half_size_);
        return (coll == 1) ? Eigen::Vector2i(i1, i2) : Eigen::Vector2i(-1, -1);
    }
};

struct convert_index_functor {
    convert_index_functor(const Eigen::Vector3i* occupied_voxels_keys, int resolution)
    : occupied_voxels_keys_(occupied_voxels_keys), resolution_(resolution) {};
    const Eigen::Vector3i* occupied_voxels_keys_;
    const int resolution_;
    __device__ Eigen::Vector2i operator()(const Eigen::Vector2i& idxs) {
        return Eigen::Vector2i(idxs[0], IndexOf(occupied_voxels_keys_[idxs[1]], resolution_));
    }
};

}  // namespace

CollisionResult::CollisionResult()
: first_(geometry::Geometry::GeometryType::Unspecified),
second_(geometry::Geometry::GeometryType::Unspecified) {};

CollisionResult::CollisionResult(geometry::Geometry::GeometryType first,
                                 geometry::Geometry::GeometryType second,
                                 const utility::device_vector<Eigen::Vector2i>& collision_index_pairs)
                                 : first_(first), second_(second),
                                 collision_index_pairs_(collision_index_pairs) {};

CollisionResult::CollisionResult(const CollisionResult& other)
: first_(other.first_), second_(other.second_), collision_index_pairs_(other.collision_index_pairs_) {};

CollisionResult::~CollisionResult() {};

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::VoxelGrid& voxelgrid1,
                                                     const geometry::VoxelGrid& voxelgrid2,
                                                     float margin) {
    auto out = std::make_shared<CollisionResult>();
    size_t n_v1 = voxelgrid1.voxels_keys_.size();
    size_t n_v2 = voxelgrid2.voxels_keys_.size();
    size_t n_total = n_v1 * n_v2;
    intersect_voxel_voxel_functor func(thrust::raw_pointer_cast(voxelgrid1.voxels_keys_.data()),
                                       thrust::raw_pointer_cast(voxelgrid2.voxels_keys_.data()),
                                       voxelgrid1.voxel_size_, voxelgrid2.voxel_size_,
                                       voxelgrid1.origin_, voxelgrid2.origin_, n_v2, margin);
    out->first_ = geometry::Geometry::GeometryType::VoxelGrid;
    out->second_ = geometry::Geometry::GeometryType::VoxelGrid;
    out->collision_index_pairs_.resize(n_total);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      out->collision_index_pairs_.begin(), func);
    remove_negative(out->collision_index_pairs_);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::VoxelGrid& voxelgrid,
                                                     const geometry::LineSet& lineset,
                                                     float margin) {
    auto out = std::make_shared<CollisionResult>();
    size_t n_v1 = voxelgrid.voxels_keys_.size();
    size_t n_v2 = lineset.lines_.size();
    size_t n_total = n_v1 * n_v2;
    intersect_voxel_line_functor func(thrust::raw_pointer_cast(voxelgrid.voxels_keys_.data()),
                                      thrust::raw_pointer_cast(lineset.points_.data()),
                                      thrust::raw_pointer_cast(lineset.lines_.data()),
                                      voxelgrid.voxel_size_, voxelgrid.origin_, n_v2, margin);
    out->first_ = geometry::Geometry::GeometryType::VoxelGrid;
    out->second_ = geometry::Geometry::GeometryType::LineSet;
    out->collision_index_pairs_.resize(n_total);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      out->collision_index_pairs_.begin(), func);
    remove_negative(out->collision_index_pairs_);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::LineSet& lineset,
                                                     const geometry::VoxelGrid& voxelgrid,
                                                     float margin) {
    auto out = ComputeIntersection(voxelgrid, lineset);
    out->first_ = geometry::Geometry::GeometryType::LineSet;
    out->second_ = geometry::Geometry::GeometryType::VoxelGrid;
    swap_index(out->collision_index_pairs_);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::VoxelGrid& voxelgrid,
                                                     const geometry::OccupancyGrid& occgrid,
                                                     float margin) {
    auto out = std::make_shared<CollisionResult>();
    size_t n_v1 = voxelgrid.voxels_keys_.size();
    auto occupied_voxels = occgrid.ExtractOccupiedVoxels();
    utility::device_vector<Eigen::Vector3i> occupied_voxels_keys(occupied_voxels->size());
    thrust::transform(occupied_voxels->begin(), occupied_voxels->end(), occupied_voxels_keys.begin(),
                      [] __device__ (const geometry::OccupancyVoxel& voxel) {
                          return voxel.grid_index_.cast<int>();
                      });
    size_t n_v2 = occupied_voxels->size();
    size_t n_total = n_v1 * n_v2;
    const Eigen::Vector3f occ_origin = occgrid.origin_ - 0.5 * occgrid.voxel_size_ * Eigen::Vector3f::Constant(occgrid.resolution_);
    intersect_voxel_voxel_functor func(thrust::raw_pointer_cast(voxelgrid.voxels_keys_.data()),
                                       thrust::raw_pointer_cast(occupied_voxels_keys.data()),
                                       voxelgrid.voxel_size_, occgrid.voxel_size_,
                                       voxelgrid.origin_, occ_origin, n_v2, margin);
    out->first_ = geometry::Geometry::GeometryType::VoxelGrid;
    out->second_ = geometry::Geometry::GeometryType::OccupancyGrid;
    out->collision_index_pairs_.resize(n_total);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      out->collision_index_pairs_.begin(), func);
    convert_index_functor func_c(thrust::raw_pointer_cast(occupied_voxels_keys.data()), occgrid.resolution_);
    thrust::transform(out->collision_index_pairs_.begin(), out->collision_index_pairs_.end(),
                      out->collision_index_pairs_.begin(), func_c);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::OccupancyGrid& occgrid,
                                                     const geometry::VoxelGrid& voxelgrid,
                                                     float margin) {
    auto out = ComputeIntersection(voxelgrid, occgrid, margin);
    out->first_ = geometry::Geometry::GeometryType::OccupancyGrid;
    out->second_ = geometry::Geometry::GeometryType::VoxelGrid;
    swap_index(out->collision_index_pairs_);
    return out;
}

}
}