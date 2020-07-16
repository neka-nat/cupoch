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
#include <thrust/gather.h>

#include "cupoch/collision/collision.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/planning/planner.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace planning {

PlannerBase& PlannerBase::AddObstacle(
        const std::shared_ptr<geometry::Geometry>& obstacle) {
    obstacles_.push_back(obstacle);
    return *this;
}

Pos3DPlanner::Pos3DPlanner(float object_radius)
    : object_radius_(object_radius) {}
Pos3DPlanner::Pos3DPlanner(const geometry::Graph<3>& graph, float object_radius)
    : graph_(graph), object_radius_(object_radius) {}

Pos3DPlanner::~Pos3DPlanner() {}

Pos3DPlanner& Pos3DPlanner::UpdateGraph() {
    for (const auto& obstacle : obstacles_) {
        auto res = std::make_shared<collision::CollisionResult>();
        switch (obstacle->GetGeometryType()) {
            case geometry::Geometry::GeometryType::VoxelGrid: {
                const geometry::VoxelGrid& voxel_grid =
                        (const geometry::VoxelGrid&)(*obstacle);
                res = collision::ComputeIntersection(voxel_grid, graph_,
                                                     object_radius_);
            }
            default: {
                utility::LogError("Unsupported obstacle type.");
            }
        }
        if (res->IsCollided()) {
            utility::device_vector<Eigen::Vector2i> remove_edges(
                    res->collision_index_pairs_.size());
            thrust::gather(thrust::make_transform_iterator(
                                   res->collision_index_pairs_.begin(),
                                   extract_element_functor<int, 2, 1>()),
                           thrust::make_transform_iterator(
                                   res->collision_index_pairs_.end(),
                                   extract_element_functor<int, 2, 1>()),
                           graph_.lines_.begin(), remove_edges.begin());
            graph_.RemoveEdges(remove_edges);
        }
    }
    return *this;
}

std::shared_ptr<Path> Pos3DPlanner::FindPath(
        const Eigen::Vector3f& start, const Eigen::Vector3f& goal) const {
    auto ex_graph = graph_;
    size_t n_start = ex_graph.points_.size();
    size_t n_goal = n_start + 1;
    ex_graph.AddNodeAndConnect(start, max_edge_distance_, false);
    ex_graph.AddNodeAndConnect(goal, max_edge_distance_, false);
    ex_graph.ConstructGraph();
    auto path_idxs = ex_graph.DijkstraPath(n_start, n_goal);
    utility::pinned_host_vector<Eigen::Vector3f> h_points = ex_graph.points_;
    auto out = std::make_shared<Path>();
    for (const auto& i : *path_idxs) {
        out->push_back(h_points[i]);
    }
    return out;
}

}  // namespace planning
}  // namespace cupoch