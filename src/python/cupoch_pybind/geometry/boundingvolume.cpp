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
#include "cupoch/geometry/boundingvolume.h"

#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

template <typename AabbT, int Dim>
void bind_axis_aligned_bounding_box(AabbT &axis_aligned_bounding_box) {
    py::detail::bind_default_constructor<geometry::AxisAlignedBoundingBox<Dim>>(
            axis_aligned_bounding_box);
    py::detail::bind_copy_functions<geometry::AxisAlignedBoundingBox<Dim>>(
            axis_aligned_bounding_box);
    axis_aligned_bounding_box
            .def(py::init<const Eigen::Matrix<float, Dim, 1> &, const Eigen::Matrix<float, Dim, 1> &>(),
                 "Create an AxisAlignedBoundingBox from min bounds and max "
                 "bounds in x, y and z",
                 "min_bound"_a, "max_bound"_a)
            .def("__repr__",
                 [](const geometry::AxisAlignedBoundingBox<Dim> &box) {
                     return std::string("geometry::AxisAlignedBoundingBox");
                 })
            .def("volume", &geometry::AxisAlignedBoundingBox<Dim>::Volume,
                 "Returns the volume of the bounding box.")
            .def("get_extent", &geometry::AxisAlignedBoundingBox<Dim>::GetExtent,
                 "Get the extent/length of the bounding box in x, y, and z "
                 "dimension.")
            .def("get_half_extent",
                 &geometry::AxisAlignedBoundingBox<Dim>::GetHalfExtent,
                 "Returns the half extent of the bounding box.")
            .def("get_max_extent",
                 &geometry::AxisAlignedBoundingBox<Dim>::GetMaxExtent,
                 "Returns the maximum extent, i.e. the maximum of X, Y and Z "
                 "axis")
            .def("get_point_indices_within_bounding_box",
                 [] (const geometry::AxisAlignedBoundingBox<Dim> &aabb,
                     const wrapper::device_vector_wrapper<Eigen::Matrix<float, Dim, 1>>& points) {
                         return wrapper::device_vector_size_t(aabb.GetPointIndicesWithinBoundingBox(points.data_));
                  },
                 "Return indices to points that are within the bounding box.",
                 "points"_a)
            .def_static(
                    "create_from_points",
                    [] (const wrapper::device_vector_wrapper<Eigen::Matrix<float, Dim, 1>>& points) {
                        return geometry::AxisAlignedBoundingBox<Dim>::CreateFromPoints(points.data_);
                    },
                    "Creates the bounding box that encloses the set of points.",
                    "points"_a)
            .def_readwrite("min_bound",
                           &geometry::AxisAlignedBoundingBox<Dim>::min_bound_,
                           "``float32`` array of shape ``(3, )``")
            .def_readwrite("max_bound",
                           &geometry::AxisAlignedBoundingBox<Dim>::max_bound_,
                           "``float32`` array of shape ``(3, )``")
            .def_readwrite("color", &geometry::AxisAlignedBoundingBox<Dim>::color_,
                           "``float32`` array of shape ``(3, )``");
}

template <typename AabbT>
void bind_axis_aligned_bounding_box3D(AabbT &axis_aligned_bounding_box) {
     axis_aligned_bounding_box
            .def("get_box_points",
                 &geometry::AxisAlignedBoundingBox<3>::template GetBoxPoints<3>,
                 "Returns the eight points that define the bounding box.")
            .def("get_print_info",
                 &geometry::AxisAlignedBoundingBox<3>::template GetPrintInfo<3>,
                 "Returns the 3D dimensions of the bounding box in string "
                 "format.");
}

void doc_inject(py::module &m, const std::string& name) {
    docstring::ClassMethodDocInject(m, name, "volume");
    docstring::ClassMethodDocInject(m, name,
                                    "get_box_points");
    docstring::ClassMethodDocInject(m, name, "get_extent");
    docstring::ClassMethodDocInject(m, name,
                                    "get_half_extent");
    docstring::ClassMethodDocInject(m, name,
                                    "get_max_extent");
    docstring::ClassMethodDocInject(m, name,
                                    "get_point_indices_within_bounding_box",
                                    {{"points", "A list of points."}});
    docstring::ClassMethodDocInject(m, name,
                                    "get_print_info");
    docstring::ClassMethodDocInject(m, name,
                                    "create_from_points",
                                    {{"points", "A list of points."}});
}

void pybind_boundingvolume(py::module &m) {
    py::class_<geometry::OrientedBoundingBox,
               PyGeometry3D<geometry::OrientedBoundingBox>,
               std::shared_ptr<geometry::OrientedBoundingBox>,
               geometry::GeometryBase3D>
            oriented_bounding_box(m, "OrientedBoundingBox",
                                  "Class that defines an oriented box that can "
                                  "be computed from 3D geometries.");
    py::detail::bind_default_constructor<geometry::OrientedBoundingBox>(
            oriented_bounding_box);
    py::detail::bind_copy_functions<geometry::OrientedBoundingBox>(
            oriented_bounding_box);
    oriented_bounding_box
            .def(py::init<const Eigen::Vector3f &, const Eigen::Matrix3f &,
                          const Eigen::Vector3f &>(),
                 "Create OrientedBoudingBox from center, rotation R and extent "
                 "in x, y and z "
                 "direction",
                 "center"_a, "R"_a, "extent"_a)
            .def("__repr__",
                 [](const geometry::OrientedBoundingBox &box) {
                     return std::string("geometry::OrientedBoundingBox");
                 })
            .def("get_point_indices_within_bounding_box",
                 [] (const geometry::OrientedBoundingBox &box,
                     const wrapper::device_vector_vector3f& points) {
                         return wrapper::device_vector_size_t(box.GetPointIndicesWithinBoundingBox(points.data_));
                  },
                 "Return indices to points that are within the bounding box.",
                 "points"_a)
            .def_static("create_from_axis_aligned_bounding_box",
                        &geometry::OrientedBoundingBox::
                                CreateFromAxisAlignedBoundingBox,
                        "Returns an oriented bounding box from the "
                        "AxisAlignedBoundingBox.",
                        "aabox"_a)
            .def("volume", &geometry::OrientedBoundingBox::Volume,
                 "Returns the volume of the bounding box.")
            .def("get_box_points", &geometry::OrientedBoundingBox::GetBoxPoints,
                 "Returns the eight points that define the bounding box.")
            .def_readwrite("center", &geometry::OrientedBoundingBox::center_,
                           "``float32`` array of shape ``(3, )``")
            .def_readwrite("R", &geometry::OrientedBoundingBox::R_,
                           "``float32`` array of shape ``(3,3 )``")
            .def_readwrite("extent", &geometry::OrientedBoundingBox::extent_,
                           "``float32`` array of shape ``(3, )``")
            .def_readwrite("color", &geometry::OrientedBoundingBox::color_,
                           "``float32`` array of shape ``(3, )``");
    docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "volume");
    docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "get_box_points");
    docstring::ClassMethodDocInject(m, "OrientedBoundingBox",
                                    "get_point_indices_within_bounding_box",
                                    {{"points", "A list of points."}});
    docstring::ClassMethodDocInject(
            m, "OrientedBoundingBox", "create_from_axis_aligned_bounding_box",
            {{"aabox",
              "AxisAlignedBoundingBox object from which OrientedBoundingBox is "
              "created."}});

    py::class_<geometry::AxisAlignedBoundingBox<3>,
               PyGeometry3D<geometry::AxisAlignedBoundingBox<3>>,
               std::shared_ptr<geometry::AxisAlignedBoundingBox<3>>,
               geometry::GeometryBase3D>
            axis_aligned_bounding_box(m, "AxisAlignedBoundingBox",
                                      "Class that defines an axis_aligned box "
                                      "that can be computed from 3D "
                                      "geometries, The axis aligned bounding "
                                      "box uses the cooridnate axes for "
                                      "bounding box generation.");
     bind_axis_aligned_bounding_box<decltype(axis_aligned_bounding_box), 3>(axis_aligned_bounding_box);
     bind_axis_aligned_bounding_box3D<decltype(axis_aligned_bounding_box)>(axis_aligned_bounding_box);
     doc_inject(m, "AxisAlignedBoundingBox");
}