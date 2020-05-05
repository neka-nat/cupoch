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
              "Add an edge to the graph", "edge"_a, "weight"_a = 1.0)
         .def("add_edges", py::overload_cast<const thrust::host_vector<Eigen::Vector2i>&, const thrust::host_vector<float>&>(&geometry::Graph::AddEdges),
              "Add edges to the graph", "edges"_a, "weights"_a = thrust::host_vector<float>())
         .def("remove_edge", &geometry::Graph::RemoveEdge,
              "Remove an edge from the graph", "edge"_a)
         .def("remove_edges", py::overload_cast<const thrust::host_vector<Eigen::Vector2i>&>(&geometry::Graph::RemoveEdges),
              "Remove edges from the graph", "edges"_a)
         .def("set_edge_weights_from_distance", &geometry::Graph::SetEdgeWeightsFromDistance)
         .def("dijkstra_path", py::overload_cast<int, int>(&geometry::Graph::DijkstraPath, py::const_))
         .def_property("edges", [] (geometry::LineSet &line) {return wrapper::device_vector_vector2i(line.lines_);},
                       [] (geometry::LineSet &line, const wrapper::device_vector_vector2i& vec) {wrapper::FromWrapper(line.lines_, vec);});
}