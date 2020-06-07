#pragma once

#include <Eigen/Core>

#include "cupoch/geometry/geometry.h"
#include "cupoch/collision/primitives.h"

namespace cupoch {

namespace geometry {
class VoxelGrid;
class LineSet;
class OccupancyGrid;
}

namespace collision {

struct CollisionResult {
    geometry::Geometry::GeometryType first_;
    geometry::Geometry::GeometryType second_;
    utility::device_vector<Eigen::Vector2i> collision_index_pairs_;

    CollisionResult();
    CollisionResult(geometry::Geometry::GeometryType first,
                    geometry::Geometry::GeometryType second,
                    const utility::device_vector<Eigen::Vector2i>& collision_index_pairs);
    CollisionResult(const CollisionResult& other);
    ~CollisionResult();

    bool IsCollided() const { return !collision_index_pairs_.empty(); };
    thrust::host_vector<Eigen::Vector2i> GetCollisionIndexPairs() const;
};

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::VoxelGrid& voxelgrid1,
                                                     const geometry::VoxelGrid& voxelgrid2,
                                                     float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::VoxelGrid& voxelgrid,
                                                     const geometry::LineSet& lineset,
                                                     float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::LineSet& lineset,
                                                     const geometry::VoxelGrid& voxelgrid,
                                                     float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::VoxelGrid& voxelgrid,
                                                     const geometry::OccupancyGrid& occgrid,
                                                     float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::OccupancyGrid& occgrid,
                                                     const geometry::VoxelGrid& voxelgrid,
                                                     float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(const PrimitiveArray& primitives,
                                                     const geometry::VoxelGrid& voxelgrid,
                                                     float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::VoxelGrid& voxelgrid,
                                                     const PrimitiveArray& primitives,
                                                     float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(const PrimitiveArray& primitives,
                                                     const geometry::OccupancyGrid& occgrid,
                                                     float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(const geometry::OccupancyGrid& occgrid,
                                                     const PrimitiveArray& primitives,
                                                     float margin = 0.0f);

std::shared_ptr<CollisionResult> ComputeIntersection(const PrimitiveArray& primitives1,
                                                     const PrimitiveArray& primitives2,
                                                     float margin = 0.0f);

}
}