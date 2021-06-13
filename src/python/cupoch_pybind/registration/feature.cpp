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
#include "cupoch/registration/feature.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/registration/registration.h"

using namespace cupoch;

namespace {
template <class FeatureT, int Dim>
void bind_def(FeatureT& feature) {
    py::detail::bind_default_constructor<registration::Feature<Dim>>(feature);
    py::detail::bind_copy_functions<registration::Feature<Dim>>(feature);
    feature.def("resize", &registration::Feature<Dim>::Resize, "n"_a,
                "Resize feature data buffer to ``33 x n``.")
            .def("dimension", &registration::Feature<Dim>::Dimension,
                 "Returns feature dimensions per point.")
            .def("num", &registration::Feature<Dim>::Num,
                 "Returns number of points.")
            .def_property(
                    "data",
                    [](registration::Feature<Dim> &ft) {
                        return wrapper::device_vector_wrapper<Eigen::Matrix<float, Dim, 1>>(ft.data_);
                    },
                    [](registration::Feature<Dim> &ft,
                       const wrapper::device_vector_wrapper<Eigen::Matrix<float, Dim, 1>> &vec) {
                        wrapper::FromWrapper(ft.data_, vec);
                    },
                    "``33 x n`` float64 numpy array: Data buffer "
                    "storing features.")
            .def("__repr__", [](const registration::Feature<Dim> &f) {
                return std::string(
                               "registration::Feature class with dimension "
                               "= ") +
                       std::to_string(f.Dimension()) +
                       std::string(" and num = ") + std::to_string(f.Num()) +
                       std::string("\nAccess its data via data member.");
            });
}

}

void pybind_feature(py::module &m) {
    py::class_<wrapper::device_vector_vector33f,
               std::shared_ptr<wrapper::device_vector_vector33f>>
            dev_vec33(m, "Vector33fVector",
                      "utility::device_vector<Eigen::Matrix<float, 33, 1>>");
    dev_vec33.def("cpu", &wrapper::device_vector_vector33f::cpu)
            .def("__len__", &wrapper::device_vector_vector33f::size);

    py::class_<wrapper::device_vector_vector352f,
               std::shared_ptr<wrapper::device_vector_vector352f>>
            dev_vec352(m, "Vector352fVector",
                      "utility::device_vector<Eigen::Matrix<float, 352, 1>>");
    dev_vec352.def("cpu", &wrapper::device_vector_vector352f::cpu)
            .def("__len__", &wrapper::device_vector_vector352f::size);

    // cupoch.registration.FeatureFPFH
    py::class_<registration::Feature<33>,
               std::shared_ptr<registration::Feature<33>>>
            feature_fpfh(m, "FeatureFPFH", "Class to store FPFH featrues for registration.");
    bind_def<decltype(feature_fpfh), 33>(feature_fpfh);

    docstring::ClassMethodDocInject(m, "FeatureFPFH", "dimension");
    docstring::ClassMethodDocInject(m, "FeatureFPFH", "num");
    docstring::ClassMethodDocInject(m, "FeatureFPFH", "resize",
                                    {{"dim", "FeatureFPFH dimension per point."},
                                     {"n", "Number of points."}});

    // cupoch.registration.FeatureSHOT
    py::class_<registration::Feature<352>,
               std::shared_ptr<registration::Feature<352>>>
            feature_shot(m, "FeatureSHOT", "Class to store SHOT featrues for registration.");
    bind_def<decltype(feature_shot), 352>(feature_shot);

    docstring::ClassMethodDocInject(m, "FeatureSHOT", "dimension");
    docstring::ClassMethodDocInject(m, "FeatureSHOT", "num");
    docstring::ClassMethodDocInject(m, "FeatureSHOT", "resize",
                                    {{"dim", "FeatureSHOT dimension per point."},
                                     {"n", "Number of points."}});
}

void pybind_feature_methods(py::module &m) {
    m.def("compute_fpfh_feature", &registration::ComputeFPFHFeature,
          "Function to compute FPFH feature for a point cloud", "input"_a,
          "search_param"_a);
    m.def("compute_shot_feature", &registration::ComputeSHOTFeature,
          "Function to compute SHOT feature for a point cloud", "input"_a,
          "radius"_a, "search_param"_a);
    docstring::FunctionDocInject(
            m, "compute_fpfh_feature",
            {{"input", "The Input point cloud."},
             {"search_param", "KDTree KNN search parameter."}});
}