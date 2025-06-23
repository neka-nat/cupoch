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
#include "cupoch/geometry/lineset.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/dl_converter.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

namespace {

template <class LineSetT, int Dim>
void bind_def(LineSetT& lineset) {
    py::detail::bind_default_constructor<geometry::LineSet<Dim>>(lineset);
    py::detail::bind_copy_functions<geometry::LineSet<Dim>>(lineset);
    lineset.def(py::init<const std::vector<Eigen::Matrix<float, Dim, 1>> &,
                         const std::vector<Eigen::Vector2i> &>(),
                "Create a LineSet from given points and line indices",
                "points"_a, "lines"_a)
            .def(py::init([](const wrapper::device_vector_wrapper<Eigen::Matrix<float, Dim, 1>> &points,
                             const wrapper::device_vector_vector2i &lines) {
                     return std::unique_ptr<geometry::LineSet<Dim>>(
                             new geometry::LineSet<Dim>(points.data_,
                                                        lines.data_));
                 }),
                 "Create a LineSet from given points and line indices",
                 "points"_a, "lines"_a)
            .def(py::init<const std::vector<Eigen::Matrix<float, Dim, 1>>>(),
                 "Create a LineSet from given path",
                 "path"_a)
            .def("__repr__",
                 [](const geometry::LineSet<Dim> &lineset) {
                     return std::string("geometry::LineSet with ") +
                            std::to_string(lineset.lines_.size()) + " lines.";
                 })
            .def("has_points", &geometry::LineSet<Dim>::HasPoints,
                 "Returns ``True`` if the object contains points.")
            .def("has_lines", &geometry::LineSet<Dim>::HasLines,
                 "Returns ``True`` if the object contains lines.")
            .def("has_colors", &geometry::LineSet<Dim>::HasColors,
                 "Returns ``True`` if the object's lines contain contain "
                 "colors.")
            .def("get_line_coordinate",
                 &geometry::LineSet<Dim>::GetLineCoordinate, "line_index"_a)
            .def("paint_uniform_color",
                 &geometry::LineSet<Dim>::PaintUniformColor,
                 "Assigns each line in the line set the same color.")
            .def("paint_indexed_color",
                 [] (geometry::LineSet<Dim>& self, const wrapper::device_vector_size_t& indices, const Eigen::Vector3f& color) {
                     return self.PaintIndexedColor(indices.data_, color);
                 })
            .def_static(
                    "create_from_point_cloud_correspondences",
                    [] (const geometry::PointCloud& cloud0, const geometry::PointCloud& cloud1, const wrapper::device_vector_wrapper<Eigen::Vector2i>& correspondences) {
                        return geometry::LineSet<Dim>::template CreateFromPointCloudCorrespondences<Dim>(cloud0, cloud1, correspondences.data_);
                    },
                    "Factory function to create a LineSet from two "
                    "pointclouds and a correspondence set.",
                    "cloud0"_a, "cloud1"_a, "correspondences"_a)
            .def_static("create_from_oriented_bounding_box",
                        &geometry::LineSet<Dim>::template CreateFromOrientedBoundingBox<Dim>,
                        "Factory function to create a LineSet from an "
                        "OrientedBoundingBox.",
                        "box"_a)
            .def_static("create_from_axis_aligned_bounding_box",
                        &geometry::LineSet<Dim>::template CreateFromAxisAlignedBoundingBox<Dim>,
                        "Factory function to create a LineSet from an "
                        "AxisAlignedBoundingBox.",
                        "box"_a)
            .def_static("create_from_triangle_mesh",
                        &geometry::LineSet<Dim>::template CreateFromTriangleMesh<Dim>,
                        "Factory function to create a LineSet from edges of a "
                        "triangle mesh.",
                        "mesh"_a)
            .def_static("create_camera_marker",
                        &geometry::LineSet<Dim>::template CreateCameraMarker<Dim>,
                        "Factory function to create a LineSet from camera parameter",
                        "intrinsic"_a, "extrinsic"_a, "marker_size"_a = 0.3)
            .def_property(
                    "points",
                    [](geometry::LineSet<Dim> &line) {
                        return wrapper::device_vector_wrapper<Eigen::Matrix<float, Dim, 1>>(line.points_);
                    },
                    [](geometry::LineSet<Dim> &line,
                       const wrapper::device_vector_wrapper<Eigen::Matrix<float, Dim, 1>> &vec) {
                        wrapper::FromWrapper(line.points_, vec);
                    })
            .def_property(
                    "lines",
                    [](geometry::LineSet<Dim> &line) {
                        return wrapper::device_vector_vector2i(line.lines_);
                    },
                    [](geometry::LineSet<Dim> &line,
                       const wrapper::device_vector_vector2i &vec) {
                        wrapper::FromWrapper(line.lines_, vec);
                    })
            .def_property(
                    "colors",
                    [](geometry::LineSet<Dim> &line) {
                        return wrapper::device_vector_vector3f(line.colors_);
                    },
                    [](geometry::LineSet<Dim> &line,
                       const wrapper::device_vector_vector3f &vec) {
                        wrapper::FromWrapper(line.colors_, vec);
                    })
            .def("to_lines_dlpack",
                 [](geometry::LineSet<Dim> &line) {
                     return dlpack::ToDLpackCapsule<Eigen::Vector2i>(line.lines_);
                 })
            .def("from_lines_dlpack",
                 [](geometry::LineSet<Dim> &line, py::capsule dlpack) {
                     dlpack::FromDLpackCapsule<Eigen::Vector2i>(dlpack, line.lines_);
                 });
}

void doc_inject(py::module &m, const std::string& name) {
    docstring::ClassMethodDocInject(m, name, "has_colors");
    docstring::ClassMethodDocInject(m, name, "has_lines");
    docstring::ClassMethodDocInject(m, name, "has_points");
    docstring::ClassMethodDocInject(m, name, "get_line_coordinate",
                                    {{"line_index", "Index of the line."}});
    docstring::ClassMethodDocInject(m, name, "paint_uniform_color",
                                    {{"color", "Color for the LineSet."}});
    docstring::ClassMethodDocInject(
            m, name, "create_from_point_cloud_correspondences",
            {{"cloud0", "First point cloud."},
             {"cloud1", "Second point cloud."},
             {"correspondences", "Set of correspondences."}});
    docstring::ClassMethodDocInject(m, name,
                                    "create_from_oriented_bounding_box",
                                    {{"box", "The input bounding box."}});
    docstring::ClassMethodDocInject(m, name,
                                    "create_from_axis_aligned_bounding_box",
                                    {{"box", "The input bounding box."}});
    docstring::ClassMethodDocInject(m, name, "create_from_triangle_mesh",
                                    {{"mesh", "The input triangle mesh."}});
}

}

void pybind_lineset(py::module &m) {
    py::class_<geometry::LineSet<3>, PyGeometry3D<geometry::LineSet<3>>,
               std::shared_ptr<geometry::LineSet<3>>, geometry::GeometryBase3D>
            lineset(m, "LineSet",
                    "LineSet define a sets of lines in 3D. A typical "
                    "application is to display the point cloud correspondence "
                    "pairs.");
    bind_def<decltype(lineset), 3>(lineset);
    doc_inject(m, "LineSet");

    py::class_<geometry::LineSet<2>, PyGeometry2D<geometry::LineSet<2>>,
               std::shared_ptr<geometry::LineSet<2>>, geometry::GeometryBase2D>
            lineset2d(m, "LineSet2D",
                      "LineSet define a sets of lines in 2D. A typical "
                      "application is to display the point cloud correspondence "
                      "pairs.");
    bind_def<decltype(lineset2d), 2>(lineset2d);
    doc_inject(m, "LineSet2D");
}

void pybind_lineset_methods(py::module &m) {}