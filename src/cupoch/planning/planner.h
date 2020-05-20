#pragma once
#include "cupoch/geometry/graph.h"
#include <vector>

namespace cupoch {
namespace planning {

class Planner {
public:
    typedef std::vector<Eigen::Vector3f> Path;

    Planner();
    Planner(const geometry::Graph& graph);
    ~Planner();

    Planner &AddObstacle(const std::shared_ptr<geometry::Geometry>& obstacle);
    Planner &UpdateGraph(float margin = 0.0f);

    std::shared_ptr<Path> FindPath(const Eigen::Vector3f& start, const Eigen::Vector3f& goal) const;
public:
    geometry::Graph graph_;

    std::vector<std::shared_ptr<geometry::Geometry>> obstacles_;
    float max_edge_distance_ = 1.0;
    float margin_ = 0.0;
};

}
}