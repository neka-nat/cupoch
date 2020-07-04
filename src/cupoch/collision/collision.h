#pragma once

#include <Eigen/Core>

#include "cupoch/collision/primitives.h"
#include "cupoch/geometry/geometry.h"

namespace cupoch {

namespace geometry {
class VoxelGrid;
template <int Dim>
class LineSet;
class OccupancyGrid;
}  // namespace geometry

namespace collision {

struct CollisionResult {
    enum class CollisionType {
        Unspecified = 0,
        Primitives = 1,
        VoxelGrid = 2,
        OccupancyGrid = 3,
        LineSet = 4,
    };
    CollisionType first_;
    CollisionType second_;
    utility::device_vector<Eigen::Vector2i> collision_index_pairs_;

    CollisionResult();
    CollisionResult(CollisionResult::CollisionType first,
                    CollisionResult::CollisionType second,
                    const utility::device_vector<Eigen::Vector2i>&
                            collision_index_pairs);
    CollisionResult(const CollisionResult& other);
    ~CollisionResult();

    bool IsCollided() const { return !collision_index_pairs_.empty(); };
    thrust::host_vector<Eigen::Vector2i> GetCollisionIndexPairs() const;
};

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::VoxelGrid& voxelgrid1,
        const geometry::VoxelGrid& voxelgrid2,
        float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::VoxelGrid& voxelgrid,
        const geometry::LineSet<3>& lineset,
        float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::LineSet<3>& lineset,
        const geometry::VoxelGrid& voxelgrid,
        float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::VoxelGrid& voxelgrid,
        const geometry::OccupancyGrid& occgrid,
        float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::OccupancyGrid& occgrid,
        const geometry::VoxelGrid& voxelgrid,
        float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(
        const PrimitiveArray& primitives,
        const geometry::VoxelGrid& voxelgrid,
        float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::VoxelGrid& voxelgrid,
        const PrimitiveArray& primitives,
        float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(
        const PrimitiveArray& primitives,
        const geometry::OccupancyGrid& occgrid,
        float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(
        const geometry::OccupancyGrid& occgrid,
        const PrimitiveArray& primitives,
        float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(
        const PrimitiveArray& primitives1,
        const PrimitiveArray& primitives2,
        float margin = 0.0f);

}  // namespace collision
}  // namespace cupoch