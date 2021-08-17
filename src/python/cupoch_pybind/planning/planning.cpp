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
#include "cupoch_pybind/planning/planning.h"

#include "cupoch/planning/planner.h"
#include "cupoch/geometry/graph.h"

using namespace cupoch;

void pybind_planning_classes(py::module &m) {
    // cupoch.registration.ICPConvergenceCriteria
    py::class_<planning::Pos3DPlanner> pos3d_planner(
            m, "Pos3DPlanner",
            "Planner implements position-based 3d path planner.");
    py::detail::bind_copy_functions<planning::Pos3DPlanner>(pos3d_planner);
    pos3d_planner
        .def(py::init([](const geometry::Graph<3>& graph,
                         float object_radius,
                         float max_edge_distance) {
                return new planning::Pos3DPlanner(graph, object_radius, max_edge_distance);
            }),
            "graph"_a, "object_radius"_a = 0.1, "max_edge_distance"_a = 1.0)
        .def("update_graph", &planning::Pos3DPlanner::UpdateGraph)
        .def("add_obstacle", &planning::Pos3DPlanner::AddObstacle)
        .def("find_path", [] (planning::Pos3DPlanner& self,
                              const Eigen::Vector3f& start,
                              const Eigen::Vector3f& goal) {
                return *self.FindPath(start, goal);
            })
        .def("get_graph", &planning::Pos3DPlanner::GetGraph)
        .def_readwrite("object_radius", &planning::Pos3DPlanner::object_radius_)
        .def_readwrite("max_edge_distance", &planning::Pos3DPlanner::max_edge_distance_);
}

void pybind_planning(py::module &m) {
    py::module m_submodule = m.def_submodule("planning");
    pybind_planning_classes(m_submodule);
}