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
#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/planning/planner.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace planning {

PlannerBase::PlannerBase(const PlannerBase& other)
    : obstacles_(other.obstacles_) {}

PlannerBase& PlannerBase::AddObstacle(
        const std::shared_ptr<geometry::Geometry>& obstacle) {
    obstacles_.push_back(obstacle);
    return *this;
}

Pos3DPlanner::Pos3DPlanner(const geometry::Graph<3>& graph,
                           float object_radius,
                           float max_edge_distance)
    : graph_(graph),
      object_radius_(object_radius),
      max_edge_distance_(max_edge_distance) {}

Pos3DPlanner::~Pos3DPlanner() {}

Pos3DPlanner::Pos3DPlanner(const Pos3DPlanner& other)
    : PlannerBase(other),
      graph_(other.graph_),
      object_radius_(other.object_radius_),
      max_edge_distance_(other.max_edge_distance_) {}

Pos3DPlanner& Pos3DPlanner::UpdateGraph() {
    RemoveCollisionEdges(graph_);
    return *this;
}

void Pos3DPlanner::RemoveCollisionEdges(geometry::Graph<3>& graph) const {
    graph.SetEdgeWeightsFromDistance();
    for (const auto& obstacle : obstacles_) {
        auto res = std::make_shared<collision::CollisionResult>();
        switch (obstacle->GetGeometryType()) {
            case geometry::Geometry::GeometryType::VoxelGrid: {
                const geometry::VoxelGrid& voxel_grid =
                        (const geometry::VoxelGrid&)(*obstacle);
                res = collision::ComputeIntersection(voxel_grid, graph,
                                                     object_radius_);
                break;
            }
            case geometry::Geometry::GeometryType::OccupancyGrid: {
                const geometry::OccupancyGrid& occ_grid =
                        (const geometry::OccupancyGrid&)(*obstacle);
                res = collision::ComputeIntersection(occ_grid, graph,
                                                     object_radius_);
                break;
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
                                   element_get_functor<Eigen::Vector2i, 1>()),
                           thrust::make_transform_iterator(
                                   res->collision_index_pairs_.end(),
                                   element_get_functor<Eigen::Vector2i, 1>()),
                           graph.lines_.begin(), remove_edges.begin());
            graph.SetEdgeWeights(remove_edges, std::numeric_limits<float>::infinity());
        }
    }
}

std::shared_ptr<Path> Pos3DPlanner::FindPath(
        const Eigen::Vector3f& start, const Eigen::Vector3f& goal) const {
    auto ex_graph = graph_;
    size_t n_start = ex_graph.points_.size();
    size_t n_goal = n_start + 1;
    ex_graph.AddNodeAndConnect(start, max_edge_distance_, true);
    ex_graph.AddNodeAndConnect(goal, max_edge_distance_, false);
    RemoveCollisionEdges(ex_graph);
    auto path_dist = ex_graph.DijkstraPath(n_start, n_goal);
    auto out = std::make_shared<Path>();
    if (std::isinf(path_dist.second)) {
        return out;
    }
    utility::pinned_host_vector<Eigen::Vector3f> h_points(
            ex_graph.points_.size());
    copy_device_to_host(ex_graph.points_, h_points);
    cudaSafeCall(cudaDeviceSynchronize());
    for (const auto& i : *path_dist.first) {
        out->push_back(h_points[i]);
    }
    return out;
}

const geometry::Graph<3>& Pos3DPlanner::GetGraph() const {
    return graph_;
}

}  // namespace planning
}  // namespace cupoch