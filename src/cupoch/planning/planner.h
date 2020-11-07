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
#include <vector>

#include "cupoch/geometry/graph.h"

namespace cupoch {
namespace planning {

typedef std::vector<Eigen::Vector3f> Path;

class PlannerBase {
public:
    PlannerBase(){};
    virtual ~PlannerBase(){};
    PlannerBase(const PlannerBase& other);

    virtual PlannerBase& AddObstacle(
            const std::shared_ptr<geometry::Geometry>& obstacle);
    virtual std::shared_ptr<Path> FindPath(
            const Eigen::Vector3f& start,
            const Eigen::Vector3f& goal) const = 0;

public:
    std::vector<std::shared_ptr<geometry::Geometry>> obstacles_;
};

class Pos3DPlanner : public PlannerBase {
public:
    Pos3DPlanner(const geometry::Graph<3>& graph,
                 float object_radius = 0.1,
                 float max_edge_distance = 1.0);
    ~Pos3DPlanner();
    Pos3DPlanner(const Pos3DPlanner& other);

    Pos3DPlanner& UpdateGraph();
    std::shared_ptr<Path> FindPath(const Eigen::Vector3f& start,
                                   const Eigen::Vector3f& goal) const;

public:
    geometry::Graph<3> graph_;

    float object_radius_;
    float max_edge_distance_;
};

}  // namespace planning
}  // namespace cupoch