#include "cupoch/collision/collision.h"
#include "cupoch/geometry/intersection_test.h"
#include "cupoch/geometry/lineset.h"
#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/geometry/voxelgrid.h"

namespace cupoch {
namespace collision {

namespace {

struct intersect_voxel_voxel_functor {
    intersect_voxel_voxel_functor(const Eigen::Vector3i* voxels_keys1,
                                  const Eigen::Vector3i* voxels_keys2,
                                  float voxel_size1,
                                  float voxel_size2,
                                  const Eigen::Vector3f& origin1,
                                  const Eigen::Vector3f& origin2,
                                  int n_v2,
                                  float margin)
        : voxels_keys1_(voxels_keys1),
          voxels_keys2_(voxels_keys2),
          voxel_size1_(voxel_size1),
          voxel_size2_(voxel_size2),
          box_half_size1_(Eigen::Vector3f(
                  voxel_size1 / 2, voxel_size1 / 2, voxel_size1 / 2)),
          box_half_size2_(Eigen::Vector3f(
                  voxel_size2 / 2, voxel_size2 / 2, voxel_size2 / 2)),
          origin1_(origin1),
          origin2_(origin2),
          n_v2_(n_v2),
          margin_(margin){};
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
    __device__ Eigen::Vector2i operator()(size_t idx) const {
        int i1 = idx / n_v2_;
        int i2 = idx % n_v2_;
        const Eigen::Vector3f h3 = Eigen::Vector3f::Constant(0.5);
        const Eigen::Vector3f ms = Eigen::Vector3f::Constant(margin_);
        Eigen::Vector3f center1 =
                ((voxels_keys1_[i1].cast<float>() + h3) * voxel_size1_) +
                origin1_;
        Eigen::Vector3f center2 =
                ((voxels_keys2_[i2].cast<float>() + h3) * voxel_size2_) +
                origin2_;
        int coll = geometry::intersection_test::AABBAABB(
                center1 - box_half_size1_ - ms, center1 + box_half_size1_ + ms,
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
                                 int n_v2,
                                 float margin)
        : voxels_keys_(voxels_keys),
          points_(points),
          lines_(lines),
          voxel_size_(voxel_size),
          box_half_size_(Eigen::Vector3f(
                  voxel_size / 2, voxel_size / 2, voxel_size / 2)),
          origin_(origin),
          n_v2_(n_v2),
          margin_(margin){};
    const Eigen::Vector3i* voxels_keys_;
    const Eigen::Vector3f* points_;
    const Eigen::Vector2i* lines_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const Eigen::Vector3f origin_;
    const int n_v2_;
    const float margin_;
    __device__ Eigen::Vector2i operator()(size_t idx) const {
        int i1 = idx / n_v2_;
        int i2 = idx % n_v2_;
        const Eigen::Vector3f h3 = Eigen::Vector3f::Constant(0.5);
        const Eigen::Vector3f ms = Eigen::Vector3f::Constant(margin_);
        Eigen::Vector3f center =
                ((voxels_keys_[i1].cast<float>() + h3) * voxel_size_) + origin_;
        Eigen::Vector2i lidx = lines_[idx];
        int coll = geometry::intersection_test::LineSegmentAABB(
                points_[lidx[0]], points_[lidx[1]], center - box_half_size_,
                center + box_half_size_);
        return (coll == 1) ? Eigen::Vector2i(i1, i2) : Eigen::Vector2i(-1, -1);
    }
};

struct intersect_primitives_voxel_functor {
    intersect_primitives_voxel_functor(const PrimitivePack* primitives,
                                       const Eigen::Vector3i* voxels_keys,
                                       float voxel_size,
                                       const Eigen::Vector3f& origin,
                                       int n_v2,
                                       float margin)
        : primitives_(primitives),
          voxels_keys_(voxels_keys),
          voxel_size_(voxel_size),
          box_half_size_(Eigen::Vector3f(
                  voxel_size / 2, voxel_size / 2, voxel_size / 2)),
          origin_(origin),
          n_v2_(n_v2),
          margin_(margin){};
    const PrimitivePack* primitives_;
    const Eigen::Vector3i* voxels_keys_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const Eigen::Vector3f origin_;
    const int n_v2_;
    const float margin_;
    __device__ Eigen::Vector2i operator()(size_t idx) const {
        int i1 = idx / n_v2_;
        int i2 = idx % n_v2_;
        const Eigen::Vector3f h3 = Eigen::Vector3f::Constant(0.5);
        const Eigen::Vector3f ms = Eigen::Vector3f::Constant(margin_);
        Eigen::Vector3f center =
                ((voxels_keys_[i2].cast<float>() + h3) * voxel_size_) + origin_;
        switch (primitives_[i1].primitive_.type_) {
            case Primitive::PrimitiveType::Box: {
                const Box box = primitives_[i1].box_;
                int coll = geometry::intersection_test::BoxBox(
                        box.lengths_ * 0.5, box.transform_.block<3, 3>(0, 0),
                        box.transform_.block<3, 1>(0, 3), box_half_size_,
                        Eigen::Matrix3f::Identity(), center);
                return (coll == 1) ? Eigen::Vector2i(i1, i2)
                                   : Eigen::Vector2i(-1, -1);
            }
            case Primitive::PrimitiveType::Sphere: {
                const Sphere sphere = primitives_[i1].sphere_;
                int coll = geometry::intersection_test::SphereAABB(
                        sphere.transform_.block<3, 1>(0, 3), sphere.radius_,
                        center - box_half_size_, center + box_half_size_);
                return (coll == 1) ? Eigen::Vector2i(i1, i2)
                                   : Eigen::Vector2i(-1, -1);
            }
            case Primitive::PrimitiveType::Capsule: {
                const Capsule capsule = primitives_[i1].capsule_;
                Eigen::Vector3f d =
                        capsule.transform_.block<3, 1>(0, 3) -
                        0.5 * capsule.height_ *
                                capsule.transform_.block<3, 1>(0, 2);
                int coll = geometry::intersection_test::CapsuleAABB(
                        capsule.radius_, d,
                        capsule.height_ * capsule.transform_.block<3, 1>(0, 2),
                        center - box_half_size_, center + box_half_size_);
                return (coll == 1) ? Eigen::Vector2i(i1, i2)
                                   : Eigen::Vector2i(-1, -1);
            }
            default: {
                return Eigen::Vector2i(-1, -1);
            }
        }
    }
};

struct convert_index_functor {
    convert_index_functor(const Eigen::Vector3i* occupied_voxels_keys,
                          int resolution)
        : occupied_voxels_keys_(occupied_voxels_keys),
          resolution_(resolution){};
    const Eigen::Vector3i* occupied_voxels_keys_;
    const int resolution_;
    __device__ Eigen::Vector2i operator()(const Eigen::Vector2i& idxs) {
        return Eigen::Vector2i(
                idxs[0], IndexOf(occupied_voxels_keys_[idxs[1]], resolution_));
    }
};

}  // namespace

CollisionResult::CollisionResult()
    : first_(CollisionResult::CollisionType::Unspecified),
      second_(CollisionResult::CollisionType::Unspecified){};

CollisionResult::CollisionResult(
        CollisionResult::CollisionType first,
        CollisionResult::CollisionType second,
        const utility::device_vector<Eigen::Vector2i>& collision_index_pairs)
    : first_(first),
      second_(second),
      collision_index_pairs_(collision_index_pairs){};

CollisionResult::CollisionResult(const CollisionResult& other)
    : first_(other.first_),
      second_(other.second_),
      collision_index_pairs_(other.collision_index_pairs_){};

CollisionResult::~CollisionResult(){};

thrust::host_vector<Eigen::Vector2i> CollisionResult::GetCollisionIndexPairs()
        const {
    thrust::host_vector<Eigen::Vector2i> h_collision_index_pairs =
            collision_index_pairs_;
    return h_collision_index_pairs;
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::VoxelGrid& voxelgrid1,
        const geometry::VoxelGrid& voxelgrid2,
        float margin) {
    auto out = std::make_shared<CollisionResult>();
    size_t n_v1 = voxelgrid1.voxels_keys_.size();
    size_t n_v2 = voxelgrid2.voxels_keys_.size();
    size_t n_total = n_v1 * n_v2;
    intersect_voxel_voxel_functor func(
            thrust::raw_pointer_cast(voxelgrid1.voxels_keys_.data()),
            thrust::raw_pointer_cast(voxelgrid2.voxels_keys_.data()),
            voxelgrid1.voxel_size_, voxelgrid2.voxel_size_, voxelgrid1.origin_,
            voxelgrid2.origin_, n_v2, margin);
    out->first_ = CollisionResult::CollisionType::VoxelGrid;
    out->second_ = CollisionResult::CollisionType::VoxelGrid;
    out->collision_index_pairs_.resize(n_total);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      out->collision_index_pairs_.begin(), func);
    remove_negative<2>(out->collision_index_pairs_);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::VoxelGrid& voxelgrid,
        const geometry::LineSet<3>& lineset,
        float margin) {
    auto out = std::make_shared<CollisionResult>();
    size_t n_v1 = voxelgrid.voxels_keys_.size();
    size_t n_v2 = lineset.lines_.size();
    size_t n_total = n_v1 * n_v2;
    intersect_voxel_line_functor func(
            thrust::raw_pointer_cast(voxelgrid.voxels_keys_.data()),
            thrust::raw_pointer_cast(lineset.points_.data()),
            thrust::raw_pointer_cast(lineset.lines_.data()),
            voxelgrid.voxel_size_, voxelgrid.origin_, n_v2, margin);
    out->first_ = CollisionResult::CollisionType::VoxelGrid;
    out->second_ = CollisionResult::CollisionType::LineSet;
    out->collision_index_pairs_.resize(n_total);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      out->collision_index_pairs_.begin(), func);
    remove_negative<2>(out->collision_index_pairs_);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::LineSet<3>& lineset,
        const geometry::VoxelGrid& voxelgrid,
        float margin) {
    auto out = ComputeIntersection(voxelgrid, lineset);
    out->first_ = CollisionResult::CollisionType::LineSet;
    out->second_ = CollisionResult::CollisionType::VoxelGrid;
    swap_index(out->collision_index_pairs_);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::VoxelGrid& voxelgrid,
        const geometry::OccupancyGrid& occgrid,
        float margin) {
    auto out = std::make_shared<CollisionResult>();
    size_t n_v1 = voxelgrid.voxels_keys_.size();
    auto occupied_voxels = occgrid.ExtractOccupiedVoxels();
    utility::device_vector<Eigen::Vector3i> occupied_voxels_keys(
            occupied_voxels->size());
    thrust::transform(occupied_voxels->begin(), occupied_voxels->end(),
                      occupied_voxels_keys.begin(),
                      [] __device__(const geometry::OccupancyVoxel& voxel) {
                          return voxel.grid_index_.cast<int>();
                      });
    size_t n_v2 = occupied_voxels->size();
    size_t n_total = n_v1 * n_v2;
    const Eigen::Vector3f occ_origin =
            occgrid.origin_ -
            0.5 * occgrid.voxel_size_ *
                    Eigen::Vector3f::Constant(occgrid.resolution_);
    intersect_voxel_voxel_functor func(
            thrust::raw_pointer_cast(voxelgrid.voxels_keys_.data()),
            thrust::raw_pointer_cast(occupied_voxels_keys.data()),
            voxelgrid.voxel_size_, occgrid.voxel_size_, voxelgrid.origin_,
            occ_origin, n_v2, margin);
    out->first_ = CollisionResult::CollisionType::VoxelGrid;
    out->second_ = CollisionResult::CollisionType::OccupancyGrid;
    out->collision_index_pairs_.resize(n_total);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      out->collision_index_pairs_.begin(), func);
    convert_index_functor func_c(
            thrust::raw_pointer_cast(occupied_voxels_keys.data()),
            occgrid.resolution_);
    thrust::transform(out->collision_index_pairs_.begin(),
                      out->collision_index_pairs_.end(),
                      out->collision_index_pairs_.begin(), func_c);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::OccupancyGrid& occgrid,
        const geometry::VoxelGrid& voxelgrid,
        float margin) {
    auto out = ComputeIntersection(voxelgrid, occgrid, margin);
    out->first_ = CollisionResult::CollisionType::OccupancyGrid;
    out->second_ = CollisionResult::CollisionType::VoxelGrid;
    swap_index(out->collision_index_pairs_);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const PrimitiveArray& primitives,
        const geometry::VoxelGrid& voxelgrid,
        float margin) {
    auto out = std::make_shared<CollisionResult>();
    size_t n_v1 = primitives.size();
    size_t n_v2 = voxelgrid.voxels_keys_.size();
    size_t n_total = n_v1 * n_v2;
    intersect_primitives_voxel_functor func(
            thrust::raw_pointer_cast(primitives.data()),
            thrust::raw_pointer_cast(voxelgrid.voxels_keys_.data()),
            voxelgrid.voxel_size_, voxelgrid.origin_, n_v2, margin);
    out->first_ = CollisionResult::CollisionType::Primitives;
    out->second_ = CollisionResult::CollisionType::VoxelGrid;
    out->collision_index_pairs_.resize(n_total);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      out->collision_index_pairs_.begin(), func);
    remove_negative<2>(out->collision_index_pairs_);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::VoxelGrid& voxelgrid,
        const PrimitiveArray& primitives,
        float margin) {
    auto out = ComputeIntersection(primitives, voxelgrid, margin);
    out->first_ = CollisionResult::CollisionType::VoxelGrid;
    out->second_ = CollisionResult::CollisionType::Primitives;
    swap_index(out->collision_index_pairs_);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const PrimitiveArray& primitives,
        const geometry::OccupancyGrid& occgrid,
        float margin) {
    auto out = std::make_shared<CollisionResult>();
    size_t n_v1 = primitives.size();
    auto occupied_voxels = occgrid.ExtractOccupiedVoxels();
    utility::device_vector<Eigen::Vector3i> occupied_voxels_keys(
            occupied_voxels->size());
    thrust::transform(occupied_voxels->begin(), occupied_voxels->end(),
                      occupied_voxels_keys.begin(),
                      [] __device__(const geometry::OccupancyVoxel& voxel) {
                          return voxel.grid_index_.cast<int>();
                      });
    size_t n_v2 = occupied_voxels->size();
    size_t n_total = n_v1 * n_v2;
    const Eigen::Vector3f occ_origin =
            occgrid.origin_ -
            0.5 * occgrid.voxel_size_ *
                    Eigen::Vector3f::Constant(occgrid.resolution_);
    intersect_primitives_voxel_functor func(
            thrust::raw_pointer_cast(primitives.data()),
            thrust::raw_pointer_cast(occupied_voxels_keys.data()),
            occgrid.voxel_size_, occ_origin, n_v2, margin);
    out->first_ = CollisionResult::CollisionType::Primitives;
    out->second_ = CollisionResult::CollisionType::OccupancyGrid;
    out->collision_index_pairs_.resize(n_total);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      out->collision_index_pairs_.begin(), func);
    convert_index_functor func_c(
            thrust::raw_pointer_cast(occupied_voxels_keys.data()),
            occgrid.resolution_);
    thrust::transform(out->collision_index_pairs_.begin(),
                      out->collision_index_pairs_.end(),
                      out->collision_index_pairs_.begin(), func_c);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::OccupancyGrid& occgrid,
        const PrimitiveArray& primitives,
        float margin) {
    auto out = ComputeIntersection(primitives, occgrid, margin);
    out->first_ = CollisionResult::CollisionType::OccupancyGrid;
    out->second_ = CollisionResult::CollisionType::Primitives;
    swap_index(out->collision_index_pairs_);
    return out;
}

}  // namespace collision
}  // namespace cupoch