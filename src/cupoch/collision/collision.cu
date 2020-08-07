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
#include <lbvh/bvh.cuh>
#include <lbvh/query.cuh>

#include "cupoch/collision/collision.h"
#include "cupoch/geometry/intersection_test.h"
#include "cupoch/geometry/lineset.h"
#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/geometry/voxelgrid.h"

namespace cupoch {
namespace collision {

namespace {


template<typename LBVHType>
struct intersect_voxel_voxel_functor {
    intersect_voxel_voxel_functor(const LBVHType& lbvh,
                                  float voxel_size,
                                  const Eigen::Vector3f& origin,
                                  float margin)
        : lbvh_(lbvh),
          voxel_size_(voxel_size),
          origin_(origin),
          margin_(margin){};
    const LBVHType lbvh_;
    const float voxel_size_;
    const Eigen::Vector3f origin_;
    const float margin_;
    __device__ Eigen::Vector2i operator()(const thrust::tuple<size_t, Eigen::Vector3i>& x) const {
        Eigen::Vector3i key = thrust::get<1>(x);
        Eigen::Vector3f vl = key.cast<float>() * voxel_size_ + origin_ - Eigen::Vector3f::Constant(margin_);
        Eigen::Vector3f vu = (key + Eigen::Vector3i::Constant(1)).cast<float>() * voxel_size_ + origin_ + Eigen::Vector3f::Constant(margin_);
        // make a query box.
        lbvh::aabb<float> box;
        box.lower = make_float4(vl[0], vl[1], vl[2], 0.0f);
        box.upper = make_float4(vu[0], vu[1], vu[2], 0.0f);
        unsigned int buffer[1];
        const auto num_found = lbvh::query_device(lbvh_, lbvh::overlaps(box), buffer, 1);
        return (num_found > 0) ? Eigen::Vector2i(buffer[0], thrust::get<0>(x)) : Eigen::Vector2i(-1, -1);
    }
};

template<typename LBVHType>
struct intersect_voxel_line_functor {
    intersect_voxel_line_functor(const LBVHType& lbvh,
                                 float voxel_size,
                                 const Eigen::Vector3f& origin,
                                 float margin)
        : lbvh_(lbvh),
          voxel_size_(voxel_size),
          box_half_size_(Eigen::Vector3f(
            voxel_size / 2, voxel_size / 2, voxel_size / 2)),
          origin_(origin),
          margin_(margin){};
    const LBVHType lbvh_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const Eigen::Vector3f origin_;
    const float margin_;
    __device__ Eigen::Vector2i operator()(const thrust::tuple<size_t, Eigen::Vector3f, Eigen::Vector3f>& x) const {
        Eigen::Vector3f p1 = thrust::get<1>(x);
        Eigen::Vector3f p2 = thrust::get<2>(x);
        const Eigen::Vector3f ms = Eigen::Vector3f::Constant(margin_);
        Eigen::Vector3f vl = p1.array().min(p2.array()).matrix() - ms;
        Eigen::Vector3f vu = p1.array().max(p2.array()).matrix() + ms;
        // make a query box.
        lbvh::aabb<float> box;
        box.lower = make_float4(vl[0], vl[1], vl[2], 0.0f);
        box.upper = make_float4(vu[0], vu[1], vu[2], 0.0f);
        unsigned int buffer[1];
        const auto num_found = lbvh::query_device(lbvh_, lbvh::overlaps(box), buffer, 1);
        if (num_found == 0) return Eigen::Vector2i(-1, -1);
        const Eigen::Vector3f h3 = Eigen::Vector3f::Constant(0.5);
        const Eigen::Vector3i& other = lbvh_.objects[buffer[0]];
        Eigen::Vector3f center =
                ((other.cast<float>() + h3) * voxel_size_) + origin_;
        int coll = geometry::intersection_test::LineSegmentAABB(
                p1, p2, center - box_half_size_,
                center + box_half_size_);
        return (coll == 1) ? Eigen::Vector2i(buffer[0], thrust::get<0>(x)) : Eigen::Vector2i(-1, -1);
    }
};

template<typename LBVHType>
struct intersect_voxel_occgrid_functor {
    intersect_voxel_occgrid_functor(const LBVHType& lbvh,
                                    float voxel_size,
                                    const Eigen::Vector3f& origin,
                                    float margin)
        : lbvh_(lbvh),
          voxel_size_(voxel_size),
          origin_(origin),
          margin_(margin){};
    const LBVHType lbvh_;
    const float voxel_size_;
    const Eigen::Vector3f origin_;
    const float margin_;
    __device__ Eigen::Vector2i operator()(const thrust::tuple<size_t, geometry::OccupancyVoxel>& x) const {
        geometry::OccupancyVoxel voxel = thrust::get<1>(x);
        Eigen::Vector3f vl = voxel.grid_index_.cast<float>() * voxel_size_ + origin_ - Eigen::Vector3f::Constant(margin_);
        Eigen::Vector3f vu = (voxel.grid_index_ + Eigen::Vector3ui16::Constant(1)).cast<float>() * voxel_size_ + origin_ + Eigen::Vector3f::Constant(margin_);
        // make a query box.
        lbvh::aabb<float> box;
        box.lower = make_float4(vl[0], vl[1], vl[2], 0.0f);
        box.upper = make_float4(vu[0], vu[1], vu[2], 0.0f);
        unsigned int buffer[1];
        const auto num_found = lbvh::query_device(lbvh_, lbvh::overlaps(box), buffer, 1);
        return (num_found > 0) ? Eigen::Vector2i(buffer[0], thrust::get<0>(x)) : Eigen::Vector2i(-1, -1);
    }
};

template<typename LBVHType>
struct intersect_voxel_primitive_functor {
    intersect_voxel_primitive_functor(const LBVHType& lbvh,
                                      float voxel_size,
                                      const Eigen::Vector3f& origin,
                                      float margin)
        : lbvh_(lbvh),
        voxel_size_(voxel_size),
        box_half_size_(Eigen::Vector3f(
          voxel_size / 2, voxel_size / 2, voxel_size / 2)),
        origin_(origin),
        margin_(margin){};
    const LBVHType lbvh_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const Eigen::Vector3f origin_;    const float margin_;
    __device__ Eigen::Vector2i operator()(const thrust::tuple<size_t, PrimitivePack>& x) const {
        PrimitivePack primitive = thrust::get<1>(x);
        const Eigen::Vector3f h3 = Eigen::Vector3f::Constant(0.5);
        unsigned int buffer[1];
        switch (primitive.primitive_.type_) {
            case Primitive::PrimitiveType::Box: {
                const Box& obox = primitive.box_;
                auto bbox = obox.GetAxisAlignedBoundingBox();
                // make a query box.
                lbvh::aabb<float> box;
                box.lower = make_float4(bbox.min_bound_[0], bbox.min_bound_[1], bbox.min_bound_[2], 0.0f);
                box.upper = make_float4(bbox.max_bound_[0], bbox.max_bound_[1], bbox.max_bound_[2], 0.0f);
                const auto num_found = lbvh::query_device(lbvh_, lbvh::overlaps(box), buffer, 1);
                if (num_found == 0) return Eigen::Vector2i(-1, -1);
                const Eigen::Vector3i& other = lbvh_.objects[buffer[0]];
                Eigen::Vector3f center =
                        ((other.cast<float>() + h3) * voxel_size_) + origin_;
                int coll = geometry::intersection_test::BoxBox(
                        obox.lengths_ * 0.5, obox.transform_.block<3, 3>(0, 0),
                        obox.transform_.block<3, 1>(0, 3), box_half_size_,
                        Eigen::Matrix3f::Identity(), center);
                return (coll == 1) ? Eigen::Vector2i(buffer[0], thrust::get<0>(x))
                                   : Eigen::Vector2i(-1, -1);
            }
            case Primitive::PrimitiveType::Sphere: {
                const Sphere& sphere = primitive.sphere_;
                auto bbox = sphere.GetAxisAlignedBoundingBox();
                lbvh::aabb<float> box;
                box.lower = make_float4(bbox.min_bound_[0], bbox.min_bound_[1], bbox.min_bound_[2], 0.0f);
                box.upper = make_float4(bbox.max_bound_[0], bbox.max_bound_[1], bbox.max_bound_[2], 0.0f);
                const auto num_found = lbvh::query_device(lbvh_, lbvh::overlaps(box), buffer, 1);
                if (num_found == 0) return Eigen::Vector2i(-1, -1);
                const Eigen::Vector3i& other = lbvh_.objects[buffer[0]];
                Eigen::Vector3f center =
                        ((other.cast<float>() + h3) * voxel_size_) + origin_;
                int coll = geometry::intersection_test::SphereAABB(
                        sphere.transform_.block<3, 1>(0, 3), sphere.radius_,
                        center - box_half_size_, center + box_half_size_);
                return (coll == 1) ? Eigen::Vector2i(buffer[0], thrust::get<0>(x))
                                   : Eigen::Vector2i(-1, -1);
            }
            case Primitive::PrimitiveType::Capsule: {
                const Capsule& capsule = primitive.capsule_;
                auto bbox = capsule.GetAxisAlignedBoundingBox();
                lbvh::aabb<float> box;
                box.lower = make_float4(bbox.min_bound_[0], bbox.min_bound_[1], bbox.min_bound_[2], 0.0f);
                box.upper = make_float4(bbox.max_bound_[0], bbox.max_bound_[1], bbox.max_bound_[2], 0.0f);
                const auto num_found = lbvh::query_device(lbvh_, lbvh::overlaps(box), buffer, 1);
                if (num_found == 0) return Eigen::Vector2i(-1, -1);
                const Eigen::Vector3i& other = lbvh_.objects[buffer[0]];
                Eigen::Vector3f center =
                        ((other.cast<float>() + h3) * voxel_size_) + origin_;
                Eigen::Vector3f d =
                        capsule.transform_.block<3, 1>(0, 3) -
                        0.5 * capsule.height_ *
                                capsule.transform_.block<3, 1>(0, 2);
                int coll = geometry::intersection_test::CapsuleAABB(
                        capsule.radius_, d,
                        capsule.height_ * capsule.transform_.block<3, 1>(0, 2),
                        center - box_half_size_, center + box_half_size_);
                return (coll == 1) ? Eigen::Vector2i(buffer[0], thrust::get<0>(x))
                                   : Eigen::Vector2i(-1, -1);
            }
            default: {
                return Eigen::Vector2i(-1, -1);
            }
        }
    }
};

template<typename LBVHType>
struct intersect_occvoxel_primitive_functor {
    intersect_occvoxel_primitive_functor(const LBVHType& lbvh,
                                         float voxel_size,
                                         const Eigen::Vector3f& origin,
                                         float margin)
        : lbvh_(lbvh),
        voxel_size_(voxel_size),
        box_half_size_(Eigen::Vector3f(
          voxel_size / 2, voxel_size / 2, voxel_size / 2)),
        origin_(origin),
        margin_(margin){};
    const LBVHType lbvh_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const Eigen::Vector3f origin_;    const float margin_;
    __device__ Eigen::Vector2i operator()(const thrust::tuple<size_t, PrimitivePack>& x) const {
        PrimitivePack primitive = thrust::get<1>(x);
        const Eigen::Vector3f h3 = Eigen::Vector3f::Constant(0.5);
        unsigned int buffer[1];
        switch (primitive.primitive_.type_) {
            case Primitive::PrimitiveType::Box: {
                const Box& obox = primitive.box_;
                auto bbox = obox.GetAxisAlignedBoundingBox();
                // make a query box.
                lbvh::aabb<float> box;
                box.lower = make_float4(bbox.min_bound_[0], bbox.min_bound_[1], bbox.min_bound_[2], 0.0f);
                box.upper = make_float4(bbox.max_bound_[0], bbox.max_bound_[1], bbox.max_bound_[2], 0.0f);
                const auto num_found = lbvh::query_device(lbvh_, lbvh::overlaps(box), buffer, 1);
                if (num_found == 0) return Eigen::Vector2i(-1, -1);
                const geometry::OccupancyVoxel& other = lbvh_.objects[buffer[0]];
                Eigen::Vector3f center =
                        ((other.grid_index_.cast<float>() + h3) * voxel_size_) + origin_;
                int coll = geometry::intersection_test::BoxBox(
                        obox.lengths_ * 0.5, obox.transform_.block<3, 3>(0, 0),
                        obox.transform_.block<3, 1>(0, 3), box_half_size_,
                        Eigen::Matrix3f::Identity(), center);
                return (coll == 1) ? Eigen::Vector2i(buffer[0], thrust::get<0>(x))
                                   : Eigen::Vector2i(-1, -1);
            }
            case Primitive::PrimitiveType::Sphere: {
                const Sphere& sphere = primitive.sphere_;
                auto bbox = sphere.GetAxisAlignedBoundingBox();
                lbvh::aabb<float> box;
                box.lower = make_float4(bbox.min_bound_[0], bbox.min_bound_[1], bbox.min_bound_[2], 0.0f);
                box.upper = make_float4(bbox.max_bound_[0], bbox.max_bound_[1], bbox.max_bound_[2], 0.0f);
                const auto num_found = lbvh::query_device(lbvh_, lbvh::overlaps(box), buffer, 1);
                if (num_found == 0) return Eigen::Vector2i(-1, -1);
                const geometry::OccupancyVoxel& other = lbvh_.objects[buffer[0]];
                Eigen::Vector3f center =
                        ((other.grid_index_.cast<float>() + h3) * voxel_size_) + origin_;
                int coll = geometry::intersection_test::SphereAABB(
                        sphere.transform_.block<3, 1>(0, 3), sphere.radius_,
                        center - box_half_size_, center + box_half_size_);
                return (coll == 1) ? Eigen::Vector2i(buffer[0], thrust::get<0>(x))
                                   : Eigen::Vector2i(-1, -1);
            }
            case Primitive::PrimitiveType::Capsule: {
                const Capsule& capsule = primitive.capsule_;
                auto bbox = capsule.GetAxisAlignedBoundingBox();
                lbvh::aabb<float> box;
                box.lower = make_float4(bbox.min_bound_[0], bbox.min_bound_[1], bbox.min_bound_[2], 0.0f);
                box.upper = make_float4(bbox.max_bound_[0], bbox.max_bound_[1], bbox.max_bound_[2], 0.0f);
                const auto num_found = lbvh::query_device(lbvh_, lbvh::overlaps(box), buffer, 1);
                if (num_found == 0) return Eigen::Vector2i(-1, -1);
                const geometry::OccupancyVoxel& other = lbvh_.objects[buffer[0]];
                Eigen::Vector3f center =
                        ((other.grid_index_.cast<float>() + h3) * voxel_size_) + origin_;
                Eigen::Vector3f d =
                        capsule.transform_.block<3, 1>(0, 3) -
                        0.5 * capsule.height_ *
                                capsule.transform_.block<3, 1>(0, 2);
                int coll = geometry::intersection_test::CapsuleAABB(
                        capsule.radius_, d,
                        capsule.height_ * capsule.transform_.block<3, 1>(0, 2),
                        center - box_half_size_, center + box_half_size_);
                return (coll == 1) ? Eigen::Vector2i(buffer[0], thrust::get<0>(x))
                                   : Eigen::Vector2i(-1, -1);
            }
            default: {
                return Eigen::Vector2i(-1, -1);
            }
        }
    }
};

struct convert_index_functor1 {
    convert_index_functor1(int resolution)
    : resolution_(resolution){};
    const int resolution_;
    __device__ Eigen::Vector2i operator()(const thrust::tuple<Eigen::Vector2i, geometry::OccupancyVoxel>& x) {
        return Eigen::Vector2i(thrust::get<0>(x)[0],
            IndexOf(thrust::get<1>(x).grid_index_.cast<int>(), resolution_));
    }
};

struct convert_index_functor2 {
    convert_index_functor2(int resolution)
    : resolution_(resolution){};
    const int resolution_;
    __device__ Eigen::Vector2i operator()(const thrust::tuple<Eigen::Vector2i, geometry::OccupancyVoxel>& x) {
        return Eigen::Vector2i(
            IndexOf(thrust::get<1>(x).grid_index_.cast<int>(), resolution_),
            thrust::get<0>(x)[0]);
    }
};

}  // namespace

CollisionResult::CollisionResult()
    : first_(CollisionResult::CollisionType::Unspecified),
      second_(CollisionResult::CollisionType::Unspecified){};

CollisionResult::CollisionResult(
        CollisionResult::CollisionType first,
        CollisionResult::CollisionType second)
    : first_(first),
      second_(second){};

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

template<>
class ConstructorImpl<geometry::VoxelGrid> {
public:
    struct aabb_getter {
        aabb_getter(float voxel_size, const Eigen::Vector3f& origin)
        : voxel_size_(voxel_size), origin_(origin) {};
        const float voxel_size_;
        const Eigen::Vector3f origin_;
        __device__ lbvh::aabb<float> operator() (const Eigen::Vector3i& obj) const {
            Eigen::Vector3f vl = obj.cast<float>() * voxel_size_ + origin_;
            Eigen::Vector3f vu = (obj + Eigen::Vector3i::Constant(1)).cast<float>() * voxel_size_ + origin_;
            lbvh::aabb<float> box;
            box.upper = make_float4(vu[0], vu[1], vu[2], 0.0f);
            box.lower = make_float4(vl[0], vl[1], vl[2], 0.0f);
            return box;
        }
    };
    ConstructorImpl(const geometry::VoxelGrid& voxelgrid)
    : bvh_(voxelgrid.voxels_keys_.begin(), voxelgrid.voxels_keys_.end(),
           aabb_getter(voxelgrid.voxel_size_, voxelgrid.origin_)) {};
    ~ConstructorImpl() {};
    lbvh::bvh<float, Eigen::Vector3i, aabb_getter> bvh_;
};

template<>
class ConstructorImpl<geometry::OccupancyGrid> {
public:
    struct aabb_getter {
        aabb_getter(float voxel_size, const Eigen::Vector3f& origin)
        : voxel_size_(voxel_size), origin_(origin) {};
        const float voxel_size_;
        const Eigen::Vector3f origin_;
        __device__ lbvh::aabb<float> operator() (const geometry::OccupancyVoxel& obj) const {
            Eigen::Vector3f vl = obj.grid_index_.cast<float>() * voxel_size_ + origin_;
            Eigen::Vector3f vu = (obj.grid_index_ + Eigen::Vector3ui16::Constant(1)).cast<float>() * voxel_size_ + origin_;
            lbvh::aabb<float> box;
            box.upper = make_float4(vu[0], vu[1], vu[2], 0.0f);
            box.lower = make_float4(vl[0], vl[1], vl[2], 0.0f);
            return box;
        }
    };
    ConstructorImpl(const utility::device_vector<geometry::OccupancyVoxel>& values,
                    float voxel_size, const Eigen::Vector3f& origin)
    : bvh_(values.begin(), values.end(),
           aabb_getter(voxel_size, origin)) {};
    ~ConstructorImpl() {};
    lbvh::bvh<float, geometry::OccupancyVoxel, aabb_getter> bvh_;
};

template <>
void Intersection<geometry::VoxelGrid>::Construct() {
    if (target_.IsEmpty()) {
        utility::LogWarning("[Intersection::Construct] target is empty.");
        return;
    }
    impl_ = std::make_shared<ConstructorImpl<geometry::VoxelGrid>>(target_);
}

template <>
void Intersection<geometry::OccupancyGrid>::Construct() {
    if (target_.IsEmpty()) {
        utility::LogWarning("[Intersection::Construct] target is empty.");
        return;
    }
    const Eigen::Vector3f occ_origin =
            target_.origin_ -
            0.5 * target_.voxel_size_ *
                    Eigen::Vector3f::Constant(target_.resolution_);
    auto occupied_voxels = target_.ExtractOccupiedVoxels();
    impl_ = std::make_shared<ConstructorImpl<geometry::OccupancyGrid>>(*occupied_voxels, target_.voxel_size_, occ_origin);
}

template <>
template <>
std::shared_ptr<CollisionResult> Intersection<geometry::VoxelGrid>::Compute<geometry::VoxelGrid>(const geometry::VoxelGrid& query, float margin) const {
    auto out = std::make_shared<CollisionResult>(CollisionResult::CollisionType::VoxelGrid,
                                                 CollisionResult::CollisionType::VoxelGrid);
    if (target_.IsEmpty() || query.IsEmpty()) {
        utility::LogWarning("[Intersection::Compute] target or query is empty.");
        return out;
    }
    const auto bvh_dev = impl_->bvh_.get_device_repr();
    intersect_voxel_voxel_functor<decltype(bvh_dev)> func(bvh_dev, query.voxel_size_, query.origin_, margin);
    out->collision_index_pairs_.resize(query.voxels_keys_.size());
    thrust::transform(
        enumerate_begin(query.voxels_keys_), enumerate_end(query.voxels_keys_),
        out->collision_index_pairs_.begin(), func);
    remove_negative<2>(out->collision_index_pairs_);
    return out;
}

template <>
template <>
std::shared_ptr<CollisionResult> Intersection<geometry::VoxelGrid>::Compute<geometry::LineSet<3>>(const geometry::LineSet<3>& query, float margin) const {
    auto out = std::make_shared<CollisionResult>(CollisionResult::CollisionType::VoxelGrid,
                                                 CollisionResult::CollisionType::LineSet);
    if (target_.IsEmpty() || query.IsEmpty()) {
        utility::LogWarning("[Intersection::Compute] target or query is empty.");
        return out;
    }
    const auto bvh_dev = impl_->bvh_.get_device_repr();
    out->collision_index_pairs_.resize(query.lines_.size());
    intersect_voxel_line_functor<decltype(bvh_dev)> func(bvh_dev,
                                                         target_.voxel_size_,
                                                         target_.origin_,
                                                         margin);
    thrust::transform(
        make_tuple_iterator(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_permutation_iterator(
                    query.points_.begin(),
                    thrust::make_transform_iterator(
                            query.lines_.begin(),
                            extract_element_functor<int, 2, 0>())),
            thrust::make_permutation_iterator(
                    query.points_.begin(),
                    thrust::make_transform_iterator(
                            query.lines_.begin(),
                            extract_element_functor<int, 2, 1>()))),
        make_tuple_iterator(
                thrust::make_counting_iterator(query.lines_.size()),
                thrust::make_permutation_iterator(
                        query.points_.begin(),
                        thrust::make_transform_iterator(
                                query.lines_.end(),
                                extract_element_functor<int, 2, 0>())),
                thrust::make_permutation_iterator(
                        query.points_.begin(),
                        thrust::make_transform_iterator(
                                query.lines_.end(),
                                extract_element_functor<int, 2, 1>()))),
        out->collision_index_pairs_.begin(), func);
    remove_negative<2>(out->collision_index_pairs_);
    return out;
}

template <>
template <>
std::shared_ptr<CollisionResult> Intersection<geometry::VoxelGrid>::Compute<geometry::OccupancyGrid>(const geometry::OccupancyGrid& query, float margin) const {
    auto out = std::make_shared<CollisionResult>(CollisionResult::CollisionType::VoxelGrid,
                                                 CollisionResult::CollisionType::OccupancyGrid);
    if (target_.IsEmpty() || query.IsEmpty()) {
        utility::LogWarning("[Intersection::Compute] target or query is empty.");
        return out;
    }
    const auto bvh_dev = impl_->bvh_.get_device_repr();
    intersect_voxel_occgrid_functor<decltype(bvh_dev)> func(bvh_dev, query.voxel_size_, query.origin_, margin);
    auto occ_voxels = query.ExtractOccupiedVoxels();
    out->collision_index_pairs_.resize(occ_voxels->size());
    thrust::transform(
        enumerate_begin(*occ_voxels), enumerate_end(*occ_voxels),
        out->collision_index_pairs_.begin(), func);
    remove_negative<2>(out->collision_index_pairs_);
    convert_index_functor1 cfunc(query.resolution_);
    thrust::transform(
            make_tuple_iterator(
                    out->collision_index_pairs_.begin(),
                    thrust::make_permutation_iterator(
                            occ_voxels->begin(),
                            thrust::make_transform_iterator(
                                    out->collision_index_pairs_.begin(),
                                    extract_element_functor<int, 2, 1>()))),
            make_tuple_iterator(
                    out->collision_index_pairs_.end(),
                    thrust::make_permutation_iterator(
                            occ_voxels->begin(),
                            thrust::make_transform_iterator(
                                    out->collision_index_pairs_.end(),
                                    extract_element_functor<int, 2, 1>()))),
            out->collision_index_pairs_.begin(), cfunc);
    return out;
}

template <>
template <>
std::shared_ptr<CollisionResult> Intersection<geometry::OccupancyGrid>::Compute<geometry::VoxelGrid>(const geometry::VoxelGrid& query, float margin) const {
    auto out = std::make_shared<CollisionResult>(CollisionResult::CollisionType::OccupancyGrid,
                                                 CollisionResult::CollisionType::VoxelGrid);
    if (target_.IsEmpty() || query.IsEmpty()) {
        utility::LogWarning("[Intersection::Compute] target or query is empty.");
        return out;
    }
    const auto bvh_dev = impl_->bvh_.get_device_repr();
    out->collision_index_pairs_.resize(query.voxels_keys_.size());
    intersect_voxel_voxel_functor<decltype(bvh_dev)> func(bvh_dev, query.voxel_size_, query.origin_, margin);
    thrust::transform(
        enumerate_begin(query.voxels_keys_), enumerate_end(query.voxels_keys_),
        out->collision_index_pairs_.begin(), func);
    remove_negative<2>(out->collision_index_pairs_);
    convert_index_functor2 cfunc(target_.resolution_);
    thrust::transform(
            thrust::device,
            make_tuple_iterator(
                    out->collision_index_pairs_.begin(),
                    thrust::make_permutation_iterator(
                            bvh_dev.objects,
                            thrust::make_transform_iterator(
                                    out->collision_index_pairs_.begin(),
                                    extract_element_functor<int, 2, 1>()))),
            make_tuple_iterator(
                    out->collision_index_pairs_.end(),
                    thrust::make_permutation_iterator(
                            bvh_dev.objects,
                            thrust::make_transform_iterator(
                                    out->collision_index_pairs_.end(),
                                    extract_element_functor<int, 2, 1>()))),
            out->collision_index_pairs_.begin(), cfunc);
    return out;
}

template <>
template <>
std::shared_ptr<CollisionResult> Intersection<geometry::VoxelGrid>::Compute<PrimitiveArray>(const PrimitiveArray& query, float margin) const {
    auto out = std::make_shared<CollisionResult>(CollisionResult::CollisionType::VoxelGrid,
                                                 CollisionResult::CollisionType::Primitives);
    if (target_.IsEmpty() || query.empty()) {
        utility::LogWarning("[Intersection::Compute] target or query is empty.");
        return out;
    }
    const auto bvh_dev = impl_->bvh_.get_device_repr();
    out->collision_index_pairs_.resize(query.size());
    intersect_voxel_primitive_functor<decltype(bvh_dev)> func(bvh_dev,
            target_.voxel_size_, target_.origin_, margin);
    thrust::transform(
        enumerate_begin(query), enumerate_end(query),
        out->collision_index_pairs_.begin(), func);
    remove_negative<2>(out->collision_index_pairs_);
    return out;
}

template <>
template <>
std::shared_ptr<CollisionResult> Intersection<geometry::OccupancyGrid>::Compute<PrimitiveArray>(const PrimitiveArray& query, float margin) const {
    auto out = std::make_shared<CollisionResult>(CollisionResult::CollisionType::OccupancyGrid,
                                                 CollisionResult::CollisionType::Primitives);
    if (target_.IsEmpty() || query.empty()) {
        utility::LogWarning("[Intersection::Compute] target or query is empty.");
        return out;
    }
    const auto bvh_dev = impl_->bvh_.get_device_repr();
    out->collision_index_pairs_.resize(query.size());
    auto occ_voxels = target_.ExtractOccupiedVoxels();
    intersect_occvoxel_primitive_functor<decltype(bvh_dev)> func(bvh_dev,
            target_.voxel_size_, target_.origin_, margin);
    thrust::transform(
        enumerate_begin(query), enumerate_end(query),
        out->collision_index_pairs_.begin(), func);
    remove_negative<2>(out->collision_index_pairs_);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::VoxelGrid& voxelgrid1,
        const geometry::VoxelGrid& voxelgrid2,
        float margin) {
    Intersection<geometry::VoxelGrid> intsct(voxelgrid1);
    return intsct.Compute<geometry::VoxelGrid>(voxelgrid2, margin);
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::VoxelGrid& voxelgrid,
        const geometry::LineSet<3>& lineset,
        float margin) {
    Intersection<geometry::VoxelGrid> intsct(voxelgrid);
    return intsct.Compute<geometry::LineSet<3>>(lineset, margin);
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
    Intersection<geometry::VoxelGrid> intsct(voxelgrid);
    return intsct.Compute<geometry::OccupancyGrid>(occgrid, margin);
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::OccupancyGrid& occgrid,
        const geometry::VoxelGrid& voxelgrid,
        float margin) {
    Intersection<geometry::OccupancyGrid> intsct(occgrid);
    return intsct.Compute<geometry::VoxelGrid>(voxelgrid, margin);
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::VoxelGrid& voxelgrid,
        const PrimitiveArray& primitives,
        float margin) {
    Intersection<geometry::VoxelGrid> intsct(voxelgrid);
    return intsct.Compute<PrimitiveArray>(primitives, margin);
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const PrimitiveArray& primitives,
        const geometry::VoxelGrid& voxelgrid,
        float margin) {
    auto out = ComputeIntersection(voxelgrid, primitives, margin);
    out->first_ = CollisionResult::CollisionType::Primitives;
    out->second_ = CollisionResult::CollisionType::VoxelGrid;
    swap_index(out->collision_index_pairs_);
    return out;
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::OccupancyGrid& occgrid,
        const PrimitiveArray& primitives,
        float margin) {
    Intersection<geometry::OccupancyGrid> intsct(occgrid);
    return intsct.Compute<PrimitiveArray>(primitives, margin);
}

std::shared_ptr<CollisionResult> ComputeIntersection(
        const PrimitiveArray& primitives,
        const geometry::OccupancyGrid& occgrid,
        float margin) {
    auto out = ComputeIntersection(occgrid, primitives, margin);
    out->first_ = CollisionResult::CollisionType::Primitives;
    out->second_ = CollisionResult::CollisionType::OccupancyGrid;
    swap_index(out->collision_index_pairs_);
    return out;
}

}  // namespace collision
}  // namespace cupoch