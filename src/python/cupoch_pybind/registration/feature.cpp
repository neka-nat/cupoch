#include "cupoch/registration/feature.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/registration/registration.h"

using namespace cupoch;

void pybind_feature(py::module &m) {
    py::class_<wrapper::device_vector_vector33f,
               std::shared_ptr<wrapper::device_vector_vector33f>>
            dev_vec33(m, "Vector33fVector",
                      "utility::device_vector<Eigen::Matrix<float, 33, 1>>");
    dev_vec33.def("cpu", &wrapper::device_vector_vector33f::cpu)
            .def("__len__", &wrapper::device_vector_vector33f::size);
    // cupoch.registration.Feature
    py::class_<registration::Feature<33>,
               std::shared_ptr<registration::Feature<33>>>
            feature(m, "Feature", "Class to store featrues for registration.");
    py::detail::bind_default_constructor<registration::Feature<33>>(feature);
    py::detail::bind_copy_functions<registration::Feature<33>>(feature);
    feature.def("resize", &registration::Feature<33>::Resize, "n"_a,
                "Resize feature data buffer to ``33 x n``.")
            .def("dimension", &registration::Feature<33>::Dimension,
                 "Returns feature dimensions per point.")
            .def("num", &registration::Feature<33>::Num,
                 "Returns number of points.")
            .def_property(
                    "data",
                    [](registration::Feature<33> &ft) {
                        return wrapper::device_vector_vector33f(ft.data_);
                    },
                    [](registration::Feature<33> &ft,
                       const wrapper::device_vector_vector33f &vec) {
                        wrapper::FromWrapper(ft.data_, vec);
                    },
                    "``33 x n`` float64 numpy array: Data buffer "
                    "storing features.")
            .def("__repr__", [](const registration::Feature<33> &f) {
                return std::string(
                               "registration::Feature class with dimension "
                               "= ") +
                       std::to_string(f.Dimension()) +
                       std::string(" and num = ") + std::to_string(f.Num()) +
                       std::string("\nAccess its data via data member.");
            });
    docstring::ClassMethodDocInject(m, "Feature", "dimension");
    docstring::ClassMethodDocInject(m, "Feature", "num");
    docstring::ClassMethodDocInject(m, "Feature", "resize",
                                    {{"dim", "Feature dimension per point."},
                                     {"n", "Number of points."}});
}

void pybind_feature_methods(py::module &m) {
    m.def("compute_fpfh_feature", &registration::ComputeFPFHFeature,
          "Function to compute FPFH feature for a point cloud", "input"_a,
          "search_param"_a);
    docstring::FunctionDocInject(
            m, "compute_fpfh_feature",
            {{"input", "The Input point cloud."},
             {"search_param", "KDTree KNN search parameter."}});
}