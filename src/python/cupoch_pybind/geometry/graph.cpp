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
         .def("add_edge", &geometry::Graph::AddEdge)
         .def("add_edges", [] (geometry::Graph& self, const thrust::host_vector<Eigen::Vector2i>& edges, const thrust::host_vector<float>& weights) {
                  return self.AddEdges(edges, weights);
              })
         .def("set_edge_weights_from_distance", &geometry::Graph::SetEdgeWeightsFromDistance);
}