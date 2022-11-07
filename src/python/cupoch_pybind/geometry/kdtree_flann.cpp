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
#include "cupoch/knn/kdtree_flann.h"

#include "cupoch/geometry/geometry.h"
#include "cupoch/geometry/geometry_utils.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"

using namespace cupoch;

void pybind_kdtreeflann(py::module &m) {
    // cupoch.geometry.KDTreeSearchParam
    py::class_<knn::KDTreeSearchParam> kdtreesearchparam(
            m, "KDTreeSearchParam", "Base class for KDTree search parameters.");
    kdtreesearchparam.def("get_search_type",
                          &knn::KDTreeSearchParam::GetSearchType,
                          "Get the search type (KNN, Radius) for the "
                          "search parameter.");

    // cupoch.geometry.KDTreeSearchParam.Type
    py::enum_<knn::KDTreeSearchParam::SearchType> kdtree_search_param_type(
            kdtreesearchparam, "Type", py::arithmetic());
    kdtree_search_param_type
            .value("KNNSearch", knn::KDTreeSearchParam::SearchType::Knn)
            .value("RadiusSearch",
                   knn::KDTreeSearchParam::SearchType::Radius)
            .export_values();

    // cupoch.geometry.KDTreeSearchParamKNN
    py::class_<knn::KDTreeSearchParamKNN> kdtreesearchparam_knn(
            m, "KDTreeSearchParamKNN", kdtreesearchparam,
            "KDTree search parameters for pure KNN search.");
    kdtreesearchparam_knn.def(py::init<int>(), "knn"_a = 30)
            .def("__repr__",
                 [](const knn::KDTreeSearchParamKNN &param) {
                     return std::string(
                                    "geometry::KDTreeSearchParamKNN with knn "
                                    "= ") +
                            std::to_string(param.knn_);
                 })
            .def_readwrite("knn", &knn::KDTreeSearchParamKNN::knn_,
                           "``knn`` neighbors will be searched.");

    // cupoch.geometry.KDTreeSearchParamRadius
    py::class_<knn::KDTreeSearchParamRadius> kdtreesearchparam_radius(
            m, "KDTreeSearchParamRadius", kdtreesearchparam,
            "KDTree search parameters for radius search.");
    kdtreesearchparam_radius.def(py::init<float, int>(), "radius"_a, "max_nn"_a)
            .def("__repr__",
                 [](const knn::KDTreeSearchParamRadius &param) {
                     return std::string(
                                    "geometry::KDTreeSearchParamRadius with "
                                    "radius = ") +
                            std::to_string(param.radius_) +
                            " and max_nn = " + std::to_string(param.max_nn_);
                 })
            .def_readwrite("radius",
                           &knn::KDTreeSearchParamRadius::radius_,
                           "Search radius.")
            .def_readwrite(
                    "max_nn", &knn::KDTreeSearchParamRadius::max_nn_,
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
    py::class_<knn::KDTreeFlann, std::shared_ptr<knn::KDTreeFlann>>
            kdtreeflann(m, "KDTreeFlann",
                        "KDTree with FLANN for nearest neighbor search.");
    kdtreeflann.def(py::init<>())
            .def(py::init([](const geometry::Geometry& geometry) {
                    return std::unique_ptr<knn::KDTreeFlann>(new knn::KDTreeFlann(geometry::ConvertVector3fVectorRef(geometry)));
                }), "geometry"_a)
            .def("set_geometry", [](knn::KDTreeFlann& self, const geometry::Geometry& geometry) {
                    self.SetRawData(geometry::ConvertVector3fVectorRef(geometry));
                 }, "geometry"_a)
            .def(
                    "search_vector_3f",
                    [](const knn::KDTreeFlann &tree,
                       const Eigen::Vector3f &query,
                       const knn::KDTreeSearchParam &param) {
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
                    [](const knn::KDTreeFlann &tree,
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
                    [](const knn::KDTreeFlann &tree,
                       const Eigen::Vector3f &query, float radius, int max_nn) {
                        thrust::host_vector<int> indices;
                        thrust::host_vector<float> distance2;
                        int k = tree.SearchRadius(query, radius, max_nn,
                                                  indices, distance2);
                        if (k < 0)
                            throw std::runtime_error(
                                    "search_radius_vector_3f() error!");
                        return std::make_tuple(k, indices, distance2);
                    },
                    "query"_a, "radius"_a, "max_nn"_a);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_radius_vector_3f",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_knn_vector_3f",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "search_vector_3f",
                                    map_kd_tree_flann_method_docs);
    docstring::ClassMethodDocInject(m, "KDTreeFlann", "set_geometry",
                                    map_kd_tree_flann_method_docs);
}