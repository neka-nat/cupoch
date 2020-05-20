#include "cupoch/planning/planner.h"
#include "cupoch/collision/collision.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/utility/console.h"

#include <thrust/gather.h>

namespace cupoch {
namespace planning {

Planner::Planner() {}
Planner::Planner(const geometry::Graph& graph)
: graph_(graph) {}

Planner::~Planner() {}

Planner &Planner::AddObstacle(const std::shared_ptr<geometry::Geometry>& obstacle) {
    obstacles_.push_back(obstacle);
    return *this;
}

Planner &Planner::UpdateGraph(float margin) {
    for (const auto& obstacle : obstacles_) {
        auto res = std::make_shared<collision::CollisionResult>();
        switch (obstacle->GetGeometryType()) {
            case geometry::Geometry::GeometryType::VoxelGrid: {
                const geometry::VoxelGrid& voxel_grid = (const geometry::VoxelGrid&)(*obstacle);
                res = collision::ComputeIntersection(voxel_grid, graph_, margin_);
            }
            default: {
                utility::LogError("Unsupported obstacle type.");
            }
        }
        if (res->IsCollided()) {
            utility::device_vector<Eigen::Vector2i> remove_edges(res->collision_index_pairs_.size());
            thrust::gather(thrust::make_transform_iterator(res->collision_index_pairs_.begin(),
                                                           extract_element_functor<int, 2, 1>()),
                           thrust::make_transform_iterator(res->collision_index_pairs_.end(),
                                                           extract_element_functor<int, 2, 1>()),
                           graph_.lines_.begin(), remove_edges.begin());
            graph_.RemoveEdges(remove_edges);
        }
    }
    return *this;
}

std::shared_ptr<Planner::Path> Planner::FindPath(const Eigen::Vector3f& start, const Eigen::Vector3f& goal) const {
    auto ex_graph = graph_;
    size_t n_start = ex_graph.points_.size();
    size_t n_goal = n_start + 1;
    ex_graph.AddNodeAndConnect(start, max_edge_distance_, false);
    ex_graph.AddNodeAndConnect(goal, max_edge_distance_, false);
    ex_graph.ConstructGraph();
    auto path_idxs = ex_graph.DijkstraPath(n_start, n_goal);
    utility::pinned_host_vector<Eigen::Vector3f> h_points = ex_graph.points_;
    auto out = std::make_shared<Planner::Path>();
    for (const auto& i : *path_idxs) {
        out->push_back(h_points[i]);
    }
    return out;
}


}
}