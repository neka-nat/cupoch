#include "cupoch/geometry/graph.h"

#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_graph(py::module &m) {
    py::class_<geometry::Graph, PyGeometry3D<geometry::Graph>,
               std::shared_ptr<geometry::Graph>, geometry::LineSet>
            graph(m, "Graph",
                  "Graph define a sets of nodes and edges in 3D.");
    py::detail::bind_default_constructor<geometry::Graph>(graph);
    py::detail::bind_copy_functions<geometry::Graph>(graph);
    graph.def(py::init<const thrust::host_vector<Eigen::Vector3f> &>(),
              "Create a Graph from given nodes and edges",
              "points"_a)
         .def("construct_graph", &geometry::Graph::ConstructGraph)
         .def("add_edge", &geometry::Graph::AddEdge,
              "Add an edge to the graph", "edge"_a, "weight"_a = 1.0, "lazy_add"_a = false)
         .def("add_edges", py::overload_cast<const thrust::host_vector<Eigen::Vector2i>&, const thrust::host_vector<float>&, bool>(&geometry::Graph::AddEdges),
              "Add edges to the graph", "edges"_a, "weights"_a = thrust::host_vector<float>(), "lazy_add"_a = false)
         .def("remove_edge", &geometry::Graph::RemoveEdge,
              "Remove an edge from the graph", "edge"_a)
         .def("remove_edges", py::overload_cast<const thrust::host_vector<Eigen::Vector2i>&>(&geometry::Graph::RemoveEdges),
              "Remove edges from the graph", "edges"_a)
         .def("paint_edge_color", &geometry::Graph::PaintEdgeColor,
              "Paint an edge with the color", "edge"_a, "color"_a)
         .def("paint_edges_color", py::overload_cast<const thrust::host_vector<Eigen::Vector2i>&, const Eigen::Vector3f&>(&geometry::Graph::PaintEdgesColor),
              "Paint edges with the color", "edges"_a, "color"_a)
         .def("paint_node_color", &geometry::Graph::PaintNodeColor,
              "Paint a node with the color", "node"_a, "color"_a)
         .def("paint_nodes_color", py::overload_cast<const thrust::host_vector<int>&, const Eigen::Vector3f&>(&geometry::Graph::PaintNodesColor),
              "Paint nodes with the color", "nodes"_a, "color"_a)
         .def("set_edge_weights_from_distance", &geometry::Graph::SetEdgeWeightsFromDistance)
         .def("dijkstra_path", [] (const geometry::Graph &graph, int start_node, int end_node) {
                  auto res = graph.DijkstraPath(start_node, end_node);
                  return *res;
              })
         .def_static("create_from_triangle_mesh",
                     &geometry::Graph::CreateFromTriangleMesh,
                     "Function to make graph from a TriangleMesh",
                     "input"_a)
         .def_static("create_from_axis_aligned_bounding_box",
                     py::overload_cast<const geometry::AxisAlignedBoundingBox&, const Eigen::Vector3i&>(&geometry::Graph::CreateFromAxisAlignedBoundingBox),
                     "Function to make graph from a AlignedBoundingBox",
                     "input"_a, "resolutions"_a)
         .def_static("create_from_axis_aligned_bounding_box",
                     py::overload_cast<const Eigen::Vector3f&, const Eigen::Vector3f&, const Eigen::Vector3i&>(&geometry::Graph::CreateFromAxisAlignedBoundingBox),
                     "Function to make graph from a AlignedBoundingBox",
                     "min_bound"_a, "max_bound"_a, "resolutions"_a)
         .def_property("edges", [] (geometry::Graph &graph) {return wrapper::device_vector_vector2i(graph.lines_);},
                       [] (geometry::Graph &graph, const wrapper::device_vector_vector2i& vec) {wrapper::FromWrapper(graph.lines_, vec);})
         .def_property("edge_weights", [] (geometry::Graph &graph) {return wrapper::device_vector_float(graph.edge_weights_);},
                       [] (geometry::Graph &graph, const wrapper::device_vector_float& vec) {wrapper::FromWrapper(graph.edge_weights_, vec);});
}