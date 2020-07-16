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
#include "cupoch_pybind/geometry/geometry.h"

#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_geometry_classes(py::module &m) {
    // cupoch.geometry functions
    m.def("get_rotation_matrix_from_xyz", &geometry::GetRotationMatrixFromXYZ,
          "rotation"_a);
    m.def("get_rotation_matrix_from_yzx", &geometry::GetRotationMatrixFromYZX,
          "rotation"_a);
    m.def("get_rotation_matrix_from_zxy", &geometry::GetRotationMatrixFromZXY,
          "rotation"_a);
    m.def("get_rotation_matrix_from_xzy", &geometry::GetRotationMatrixFromXZY,
          "rotation"_a);
    m.def("get_rotation_matrix_from_zyx", &geometry::GetRotationMatrixFromZYX,
          "rotation"_a);
    m.def("get_rotation_matrix_from_yxz", &geometry::GetRotationMatrixFromYXZ,
          "rotation"_a);
    m.def("get_rotation_matrix_from_axis_angle",
          &geometry::GetRotationMatrixFromAxisAngle, "rotation"_a);
    m.def("get_rotation_matrix_from_quaternion",
          &geometry::GetRotationMatrixFromQuaternion, "rotation"_a);

    // cupoch.geometry.Geometry
    py::class_<geometry::Geometry, PyGeometry<geometry::Geometry>,
               std::shared_ptr<geometry::Geometry>>
            geometry(m, "Geometry", "The base geometry class.");
    geometry.def("clear", &geometry::Geometry::Clear,
                 "Clear all elements in the geometry.")
            .def("is_empty", &geometry::Geometry::IsEmpty,
                 "Returns ``True`` iff the geometry is empty.")
            .def("get_geometry_type", &geometry::Geometry::GetGeometryType,
                 "Returns one of registered geometry types.")
            .def("dimension", &geometry::Geometry::Dimension,
                 "Returns whether the geometry is 2D or 3D.");
    docstring::ClassMethodDocInject(m, "Geometry", "clear");
    docstring::ClassMethodDocInject(m, "Geometry", "is_empty");
    docstring::ClassMethodDocInject(m, "Geometry", "get_geometry_type");
    docstring::ClassMethodDocInject(m, "Geometry", "dimension");

    // cupoch.geometry.Geometry.Type
    py::enum_<geometry::Geometry::GeometryType> geometry_type(geometry, "Type",
                                                              py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    geometry_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Geometry types.";
            }),
            py::none(), py::none(), "");

    geometry_type
            .value("Unspecified", geometry::Geometry::GeometryType::Unspecified)
            .value("PointCloud", geometry::Geometry::GeometryType::PointCloud)
            .value("VoxelGrid", geometry::Geometry::GeometryType::VoxelGrid)
            .value("OccupancyGrid",
                   geometry::Geometry::GeometryType::OccupancyGrid)
            .value("LineSet", geometry::Geometry::GeometryType::LineSet)
            .value("TriangleMesh",
                   geometry::Geometry::GeometryType::TriangleMesh)
            .value("Image", geometry::Geometry::GeometryType::Image)
            .value("RGBDImage", geometry::Geometry::GeometryType::RGBDImage)
            .export_values();

    py::class_<geometry::GeometryBase<3>,
               PyGeometry3D<geometry::GeometryBase<3>>,
               std::shared_ptr<geometry::GeometryBase<3>>, geometry::Geometry>
            geometry3d(m, "Geometry3D",
                       "The base geometry class for 3D geometries.");
    geometry3d
            .def("get_min_bound", &geometry::GeometryBase<3>::GetMinBound,
                 "Returns min bounds for geometry coordinates.")
            .def("get_max_bound", &geometry::GeometryBase<3>::GetMaxBound,
                 "Returns max bounds for geometry coordinates.")
            .def("get_center", &geometry::GeometryBase<3>::GetCenter,
                 "Returns the center of the geometry coordinates.")
            .def("get_axis_aligned_bounding_box",
                 &geometry::GeometryBase<3>::GetAxisAlignedBoundingBox,
                 "Returns an axis-aligned bounding box of the geometry.")
            .def("transform", &geometry::GeometryBase<3>::Transform,
                 "Apply transformation (4x4 matrix) to the geometry "
                 "coordinates.")
            .def("translate", &geometry::GeometryBase<3>::Translate,
                 "Apply translation to the geometry coordinates.",
                 "translation"_a, "relative"_a = true)
            .def("scale", &geometry::GeometryBase<3>::Scale,
                 "Apply scaling to the geometry coordinates.", "scale"_a,
                 "center"_a = true)
            .def("rotate", &geometry::GeometryBase<3>::Rotate,
                 "Apply rotation to the geometry coordinates and normals.",
                 "R"_a, "center"_a = true);
    docstring::ClassMethodDocInject(m, "Geometry3D", "get_min_bound");
    docstring::ClassMethodDocInject(m, "Geometry3D", "get_max_bound");
    docstring::ClassMethodDocInject(m, "Geometry3D", "get_center");
    docstring::ClassMethodDocInject(m, "Geometry3D",
                                    "get_axis_aligned_bounding_box");
    docstring::ClassMethodDocInject(m, "Geometry3D", "transform");
    docstring::ClassMethodDocInject(
            m, "Geometry3D", "translate",
            {{"translation", "A 3D vector to transform the geometry"},
             {"relative",
              "If true, the translation vector is directly added to the "
              "geometry "
              "coordinates. Otherwise, the center is moved to the translation "
              "vector."}});
    docstring::ClassMethodDocInject(
            m, "Geometry3D", "scale",
            {{"scale",
              "The scale parameter that is multiplied to the points/vertices "
              "of the geometry"},
             {"center",
              "If true, then the scale is applied to the centered geometry"}});
    docstring::ClassMethodDocInject(m, "Geometry3D", "rotate",
                                    {{"R", "The rotation matrix"},
                                     {"center",
                                      "If true, then the rotation is applied "
                                      "to the centered geometry"}});

    // cupoch.geometry.Geometry2D
    py::class_<geometry::GeometryBase<2>,
               PyGeometry2D<geometry::GeometryBase<2>>,
               std::shared_ptr<geometry::GeometryBase<2>>, geometry::Geometry>
            geometry2d(m, "Geometry2D",
                       "The base geometry class for 2D geometries.");
    geometry2d
            .def("get_min_bound", &geometry::GeometryBase<2>::GetMinBound,
                 "Returns min bounds for geometry coordinates.")
            .def("get_max_bound", &geometry::GeometryBase<2>::GetMaxBound,
                 "Returns max bounds for geometry coordinates.");
    docstring::ClassMethodDocInject(m, "Geometry2D", "get_min_bound");
    docstring::ClassMethodDocInject(m, "Geometry2D", "get_max_bound");
}

void pybind_geometry(py::module &m) {
    py::module m_submodule = m.def_submodule("geometry");
    pybind_geometry_classes(m_submodule);
    pybind_kdtreeflann(m_submodule);
    pybind_pointcloud(m_submodule);
    pybind_voxelgrid(m_submodule);
    pybind_occupanygrid(m_submodule);
    pybind_laserscanbuffer(m_submodule);
    pybind_lineset(m_submodule);
    pybind_graph(m_submodule);
    pybind_meshbase(m_submodule);
    pybind_trianglemesh(m_submodule);
    pybind_image(m_submodule);
    pybind_boundingvolume(m_submodule);
}