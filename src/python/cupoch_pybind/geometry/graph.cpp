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
#include "cupoch/geometry/graph.h"

#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/dl_converter.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_graph(py::module &m) {
    py::class_<geometry::Graph<3>, PyGeometry3D<geometry::Graph<3>>,
               std::shared_ptr<geometry::Graph<3>>, geometry::LineSet<3>>
            graph(m, "Graph", "Graph define a sets of nodes and edges in 3D.");
    py::detail::bind_default_constructor<geometry::Graph<3>>(graph);
    py::detail::bind_copy_functions<geometry::Graph<3>>(graph);
    graph.def(py::init<const thrust::host_vector<Eigen::Vector3f> &>(),
              "Create a Graph from given nodes and edges", "points"_a)
            .def("construct_graph", &geometry::Graph<3>::ConstructGraph,
                 "Construct graph structure, when given nodes and edges.",
                 "set_edge_weights_from_distance"_a = true)
            .def("add_edge", &geometry::Graph<3>::AddEdge,
                 "Add an edge to the graph", "edge"_a, "weight"_a = 1.0,
                 "lazy_add"_a = false)
            .def("add_edges",
                 py::overload_cast<const thrust::host_vector<Eigen::Vector2i> &,
                                   const thrust::host_vector<float> &, bool>(
                         &geometry::Graph<3>::AddEdges),
                 "Add edges to the graph", "edges"_a,
                 "weights"_a = thrust::host_vector<float>(),
                 "lazy_add"_a = false)
            .def("remove_edge", &geometry::Graph<3>::RemoveEdge,
                 "Remove an edge from the graph", "edge"_a)
            .def("remove_edges",
                 py::overload_cast<const thrust::host_vector<Eigen::Vector2i>
                                           &>(&geometry::Graph<3>::RemoveEdges),
                 "Remove edges from the graph", "edges"_a)
            .def("paint_edge_color", &geometry::Graph<3>::PaintEdgeColor,
                 "Paint an edge with the color", "edge"_a, "color"_a)
            .def("paint_edges_color",
                 py::overload_cast<const thrust::host_vector<Eigen::Vector2i> &,
                                   const Eigen::Vector3f &>(
                         &geometry::Graph<3>::PaintEdgesColor),
                 "Paint edges with the color", "edges"_a, "color"_a)
            .def("paint_node_color", &geometry::Graph<3>::PaintNodeColor,
                 "Paint a node with the color", "node"_a, "color"_a)
            .def("paint_nodes_color",
                 py::overload_cast<const thrust::host_vector<int> &,
                                   const Eigen::Vector3f &>(
                         &geometry::Graph<3>::PaintNodesColor),
                 "Paint nodes with the color", "nodes"_a, "color"_a)
            .def("set_edge_weights_from_distance",
                 &geometry::Graph<3>::SetEdgeWeightsFromDistance)
            .def("dijkstra_path",
                 [](const geometry::Graph<3> &graph, int start_node,
                    int end_node) {
                     auto res = graph.DijkstraPath(start_node, end_node);
                     return *res;
                 })
            .def_static("create_from_triangle_mesh",
                        &geometry::Graph<3>::CreateFromTriangleMesh,
                        "Function to make graph from a TriangleMesh", "input"_a)
            .def_static(
                    "create_from_axis_aligned_bounding_box",
                    py::overload_cast<const geometry::AxisAlignedBoundingBox &,
                                      const Eigen::Vector3i &>(
                            &geometry::Graph<
                                    3>::CreateFromAxisAlignedBoundingBox),
                    "Function to make graph from a AlignedBoundingBox",
                    "input"_a, "resolutions"_a)
            .def_static("create_from_axis_aligned_bounding_box",
                        py::overload_cast<const Eigen::Vector3f &,
                                          const Eigen::Vector3f &,
                                          const Eigen::Vector3i &>(
                                &geometry::Graph<
                                        3>::CreateFromAxisAlignedBoundingBox),
                        "Function to make graph from a AlignedBoundingBox",
                        "min_bound"_a, "max_bound"_a, "resolutions"_a)
            .def_property(
                    "edges",
                    [](geometry::Graph<3> &graph) {
                        return wrapper::device_vector_vector2i(graph.lines_);
                    },
                    [](geometry::Graph<3> &graph,
                       const wrapper::device_vector_vector2i &vec) {
                        wrapper::FromWrapper(graph.lines_, vec);
                    })
            .def_property(
                    "edge_weights",
                    [](geometry::Graph<3> &graph) {
                        return wrapper::device_vector_float(
                                graph.edge_weights_);
                    },
                    [](geometry::Graph<3> &graph,
                       const wrapper::device_vector_float &vec) {
                        wrapper::FromWrapper(graph.edge_weights_, vec);
                    })
            .def("to_edges_dlpack",
                 [](geometry::Graph<3> &graph) {
                     return dlpack::ToDLpackCapsule<Eigen::Vector2i>(graph.lines_);
                 })
            .def("from_edges_dlpack",
                 [](geometry::Graph<3> &graph, py::capsule dlpack) {
                     dlpack::FromDLpackCapsule<Eigen::Vector2i>(dlpack, graph.lines_);
                 });
}