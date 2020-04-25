#include "cupoch/geometry/lineset.h"
#include "cupoch/geometry/pointcloud.h"

#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_lineset(py::module &m) {
    py::class_<geometry::LineSet, PyGeometry3D<geometry::LineSet>,
               std::shared_ptr<geometry::LineSet>, geometry::Geometry3D>
            lineset(m, "LineSet",
                    "LineSet define a sets of lines in 3D. A typical "
                    "application is to display the point cloud correspondence "
                    "pairs.");
    py::detail::bind_default_constructor<geometry::LineSet>(lineset);
    py::detail::bind_copy_functions<geometry::LineSet>(lineset);
    lineset.def(py::init<const thrust::host_vector<Eigen::Vector3f> &,
                         const thrust::host_vector<Eigen::Vector2i> &>(),
                "Create a LineSet from given points and line indices",
                "points"_a, "lines"_a)
            .def(py::init([](const wrapper::device_vector_vector3f& points,
                             const wrapper::device_vector_vector2i& lines) {
                    return std::unique_ptr<geometry::LineSet>(new geometry::LineSet(points.data_, lines.data_));
               }), "Create a LineSet from given points and line indices",
               "points"_a, "lines"_a)
            .def("__repr__",
                 [](const geometry::LineSet &lineset) {
                     return std::string("geometry::LineSet with ") +
                            std::to_string(lineset.lines_.size()) + " lines.";
                 })
            .def("has_points", &geometry::LineSet::HasPoints,
                 "Returns ``True`` if the object contains points.")
            .def("has_lines", &geometry::LineSet::HasLines,
                 "Returns ``True`` if the object contains lines.")
            .def("has_colors", &geometry::LineSet::HasColors,
                 "Returns ``True`` if the object's lines contain contain "
                 "colors.")
            .def("get_line_coordinate", &geometry::LineSet::GetLineCoordinate,
                 "line_index"_a)
            .def("paint_uniform_color", &geometry::LineSet::PaintUniformColor,
                 "Assigns each line in the line set the same color.")
            .def_static("create_from_point_cloud_correspondences",
                        &geometry::LineSet::CreateFromPointCloudCorrespondences,
                        "Factory function to create a LineSet from two "
                        "pointclouds and a correspondence set.",
                        "cloud0"_a, "cloud1"_a, "correspondences"_a)
            .def_static("create_from_oriented_bounding_box",
                        &geometry::LineSet::CreateFromOrientedBoundingBox,
                        "Factory function to create a LineSet from an "
                        "OrientedBoundingBox.",
                        "box"_a)
            .def_static("create_from_axis_aligned_bounding_box",
                        &geometry::LineSet::CreateFromAxisAlignedBoundingBox,
                        "Factory function to create a LineSet from an "
                        "AxisAlignedBoundingBox.",
                        "box"_a)
            .def_static("create_from_triangle_mesh",
                        &geometry::LineSet::CreateFromTriangleMesh,
                        "Factory function to create a LineSet from edges of a "
                        "triangle mesh.",
                        "mesh"_a)
            .def_property("points", [] (geometry::LineSet &line) {return wrapper::device_vector_vector3f(line.points_);},
                                    [] (geometry::LineSet &line, const wrapper::device_vector_vector3f& vec) {wrapper::FromWrapper(line.points_, vec);})
            .def_property("lines", [] (geometry::LineSet &line) {return wrapper::device_vector_vector2i(line.lines_);},
                                   [] (geometry::LineSet &line, const wrapper::device_vector_vector2i& vec) {wrapper::FromWrapper(line.lines_, vec);})
            .def_property("colors", [] (geometry::LineSet &line) {return wrapper::device_vector_vector3f(line.colors_);},
                                    [] (geometry::LineSet &line, const wrapper::device_vector_vector3f& vec) {wrapper::FromWrapper(line.colors_, vec);});
    docstring::ClassMethodDocInject(m, "LineSet", "has_colors");
    docstring::ClassMethodDocInject(m, "LineSet", "has_lines");
    docstring::ClassMethodDocInject(m, "LineSet", "has_points");
    docstring::ClassMethodDocInject(m, "LineSet", "get_line_coordinate",
                                    {{"line_index", "Index of the line."}});
    docstring::ClassMethodDocInject(m, "LineSet", "paint_uniform_color",
                                    {{"color", "Color for the LineSet."}});
    docstring::ClassMethodDocInject(
            m, "LineSet", "create_from_point_cloud_correspondences",
            {{"cloud0", "First point cloud."},
             {"cloud1", "Second point cloud."},
             {"correspondences", "Set of correspondences."}});
    docstring::ClassMethodDocInject(m, "LineSet",
                                    "create_from_oriented_bounding_box",
                                    {{"box", "The input bounding box."}});
    docstring::ClassMethodDocInject(m, "LineSet",
                                    "create_from_axis_aligned_bounding_box",
                                    {{"box", "The input bounding box."}});
    docstring::ClassMethodDocInject(m, "LineSet", "create_from_triangle_mesh",
                                    {{"mesh", "The input triangle mesh."}});
}

void pybind_lineset_methods(py::module &m) {}