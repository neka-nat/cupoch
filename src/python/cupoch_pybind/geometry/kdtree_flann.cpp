#include "cupoch/geometry/kdtree_flann.h"

#include "cupoch/geometry/geometry.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"

using namespace cupoch;

void pybind_kdtreeflann(py::module &m) {
    // cupoch.geometry.KDTreeSearchParam
    py::class_<geometry::KDTreeSearchParam> kdtreesearchparam(
            m, "KDTreeSearchParam", "Base class for KDTree search parameters.");
    kdtreesearchparam.def("get_search_type",
                          &geometry::KDTreeSearchParam::GetSearchType,
                          "Get the search type (KNN, Radius, Hybrid) for the "
                          "search parameter.");

    // cupoch.geometry.KDTreeSearchParam.Type
    py::enum_<geometry::KDTreeSearchParam::SearchType> kdtree_search_param_type(
            kdtreesearchparam, "Type", py::arithmetic());
    kdtree_search_param_type
            .value("KNNSearch", geometry::KDTreeSearchParam::SearchType::Knn)
            .value("RadiusSearch",
                   geometry::KDTreeSearchParam::SearchType::Radius)
            .value("HybridSearch",
                   geometry::KDTreeSearchParam::SearchType::Hybrid)
            .export_values();

    // cupoch.geometry.KDTreeSearchParamKNN
    py::class_<geometry::KDTreeSearchParamKNN> kdtreesearchparam_knn(
            m, "KDTreeSearchParamKNN", kdtreesearchparam,
            "KDTree search parameters for pure KNN search.");
    kdtreesearchparam_knn.def(py::init<int>(), "knn"_a = 30)
            .def("__repr__",
                 [](const geometry::KDTreeSearchParamKNN &param) {
                     return std::string(
                                    "geometry::KDTreeSearchParamKNN with knn "
                                    "= ") +
                            std::to_string(param.knn_);
                 })
            .def_readwrite("knn", &geometry::KDTreeSearchParamKNN::knn_,
                           "``knn`` neighbors will be searched.");

    // cupoch.geometry.KDTreeSearchParamRadius
    py::class_<geometry::KDTreeSearchParamRadius> kdtreesearchparam_radius(
            m, "KDTreeSearchParamRadius", kdtreesearchparam,
            "KDTree search parameters for pure radius search.");
    kdtreesearchparam_radius.def(py::init<float>(), "radius"_a)
            .def("__repr__",
                 [](const geometry::KDTreeSearchParamRadius &param) {
                     return std::string(
                                    "geometry::KDTreeSearchParamRadius with "
                                    "radius = ") +
                            std::to_string(param.radius_);
                 })
            .def_readwrite("radius",
                           &geometry::KDTreeSearchParamRadius::radius_,
                           "Search radius.");

    // cupoch.geometry.KDTreeSearchParamHybrid
    py::class_<geometry::KDTreeSearchParamHybrid> kdtreesearchparam_hybrid(
            m, "KDTreeSearchParamHybrid", kdtreesearchparam,
            "KDTree search parameters for hybrid KNN and radius search.");
    kdtreesearchparam_hybrid.def(py::init<float, int>(), "radius"_a, "max_nn"_a)
            .def("__repr__",
                 [](const geometry::KDTreeSearchParamHybrid &param) {
                     return std::string(
                                    "geometry::KDTreeSearchParamHybrid with "
                                    "radius = ") +
                            std::to_string(param.radius_) +
                            " and max_nn = " + std::to_string(param.max_nn_);
                 })
            .def_readwrite("radius",
                           &geometry::KDTreeSearchParamHybrid::radius_,
                           "Search radius.")
            .def_readwrite(
                    "max_nn", &geometry::KDTreeSearchParamHybrid::max_nn_,
                    "At maximum, ``max_nn`` neighbors will be searched.");

    // cupoch.geometry.KDTreeFlann
    static const std::unordered_map<std::string, std::string>
            map_kd_tree_flann_method_docs = {
                    {"query", "The input query point."},
                    {"radius", "Search radius."},
                    {"max_nn",
                     "At maximum, ``max_nn`` neighbors will be searched."},
                    {"knn", "``knn`` neighbors will be searched."},
                    {"feature", "Feature data."},
                    {"data", "Matrix data."}};
    py::class_<geometry::KDTreeFlann, std::shared_ptr<geometry::KDTreeFlann>>
            kdtreeflann(m, "KDTreeFlann",
                        "KDTree with FLANN for nearest neighbor search.");
    kdtreeflann.def(py::init<>())
            .def(py::init<const geometry::Geometry &>(), "geometry"_a)
            .def("set_geometry", &geometry::KDTreeFlann::SetGeometry,
                 "geometry"_a)
            .def(
                    "search_vector_3f",
                    [](const geometry::KDTreeFlann &tree,
                       const Eigen::Vector3f &query,
                       const geometry::KDTreeSearchParam &param) {
                        thrust::host_vector<int> indices;
                        thrust::host_vector<float> distance2;
                        int k = tree.Search(query, param, indices, distance2);
                        if (k < 0)
                            throw std::runtime_error(
                                    "search_vector_3f() error!");
                        return std::make_tuple(k, indices, distance2);
                    },
                    "query"_a, "search_param"_a)
            .def(
                    "search_knn_vector_3f",
                    [](const geometry::KDTreeFlann &tree,
                       const Eigen::Vector3f &query, int knn) {
                        thrust::host_vector<int> indices;
                        thrust::host_vector<float> distance2;
                        int k = tree.SearchKNN(query, knn, indices, distance2);
                        if (k < 0)
                            throw std::runtime_error(
                                    "search_knn_vector_3f() error!");
                        return std::make_tuple(k, indices, distance2);
                    },
                    "query"_a, "knn"_a)
            .def(
                    "search_radius_vector_3f",
                    [](const geometry::KDTreeFlann &tree,
                       const Eigen::Vector3f &query, float radius) {
                        thrust::host_vector<int> indices;
                        thrust::host_vector<float> distance2;
                        int k = tree.SearchRadius(query, radius, indices,
                                                  distance2);
                        if (k < 0)
                            throw std::runtime_error(
                                    "search_radius_vector_3f() error!");
                        return std::make_tuple(k, indices, distance2);
                    },
                    "query"_a, "radius"_a)
            .def(
                    "search_hybrid_vector_3f",
                    [](const geometry::KDTreeFlann &tree,
                       const Eigen::Vector3f &query, float radius, int max_nn) {
                        thrust::host_vector<int> indices;
                        thrust::host_vector<float> distance2;
                        int k = tree.SearchHybrid(query, radius, max_nn,
                                                  indices, distance2);
                        if (k < 0)
                            throw std::runtime_error(
                                    "search_hybrid_vector_3f() error!");
                        return std::make_tuple(k, indices, distance2);
                    },
                    "query"_a, "radius"_a, "max_nn"_a);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_hybrid_vector_3f",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_knn_vector_3f",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_radius_vector_3f",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_vector_3f",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "set_geometry",
                                    map_kd_tree_flann_method_docs);
}