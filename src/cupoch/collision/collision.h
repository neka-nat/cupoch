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
                    CollisionResult::CollisionType second);
    CollisionResult(CollisionResult::CollisionType first,
                    CollisionResult::CollisionType second,
                    const utility::device_vector<Eigen::Vector2i>&
                            collision_index_pairs);
    CollisionResult(const CollisionResult& other);
    ~CollisionResult();

    bool IsCollided() const { return !collision_index_pairs_.empty(); };
    thrust::host_vector<Eigen::Vector2i> GetCollisionIndexPairs() const;
};

template <typename TargetT>
class ConstructorImpl;

template <typename TargetT>
class Intersection {
public:
    Intersection(const TargetT& target) : target_(target) { Construct(); };
    ~Intersection() {};

    template <typename QueryT>
    std::shared_ptr<CollisionResult> Compute(const QueryT& query, float margin = 0.0f) const;
private:
    void Construct();
public:
    const TargetT& target_;
    std::shared_ptr<ConstructorImpl<TargetT>> impl_;
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