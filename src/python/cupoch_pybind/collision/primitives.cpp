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
#include "cupoch/collision/primitives.h"

#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch_pybind/collision/collision.h"
#include "cupoch_pybind/docstring.h"

using namespace cupoch;
using namespace cupoch::collision;

void pybind_primitives(py::module &m) {
    py::class_<Primitive, std::shared_ptr<Primitive>> primitive(
            m, "Primitive", "Primitive shape class.");
    py::detail::bind_default_constructor<Primitive>(primitive);
    py::detail::bind_copy_functions<Primitive>(primitive);
    primitive
            .def("get_axis_aligned_bounding_box",
                 &Primitive::GetAxisAlignedBoundingBox)
            .def_readwrite("transform", &Primitive::transform_);
    py::enum_<Primitive::PrimitiveType> primitive_type(primitive, "Type",
                                                       py::arithmetic());
    primitive_type.value("Unspecified", Primitive::PrimitiveType::Unspecified)
            .value("Box", Primitive::PrimitiveType::Box)
            .value("Sphere", Primitive::PrimitiveType::Sphere)
            .value("Capsule", Primitive::PrimitiveType::Capsule)
            .export_values();

    py::class_<Box, std::shared_ptr<Box>, Primitive> box(
            m, "Box",
            "Box class. A box consists of a center point "
            "coordinate, and lengths.");
    py::detail::bind_default_constructor<Box>(box);
    py::detail::bind_copy_functions<Box>(box);
    box.def(py::init<const Eigen::Vector3f &>(), "Create a Box", "lengths"_a)
            .def(py::init<const Eigen::Vector3f &, const Eigen::Matrix4f &>(),
                 "Create a Box", "lengths"_a, "transform"_a)
            .def_readwrite("lengths", &Box::lengths_);

    py::class_<Sphere, std::shared_ptr<Sphere>, Primitive> sphere(
            m, "Sphere",
            "Sphere class. A sphere consists of a center point "
            "coordinate, and radius.");
    py::detail::bind_default_constructor<Sphere>(sphere);
    py::detail::bind_copy_functions<Sphere>(sphere);
    sphere.def(py::init<float>(), "Create a Sphere", "radius"_a)
            .def(py::init<float, const Eigen::Vector3f &>(), "Create a Sphere",
                 "radius"_a, "center"_a)
            .def_readwrite("radius", &Sphere::radius_);

    py::class_<Capsule, std::shared_ptr<Capsule>, Primitive> capsule(
            m, "Capsule",
            "Capsule class. A Capsule consists of a transformation, "
            "radius, and height.");
    py::detail::bind_default_constructor<Capsule>(capsule);
    py::detail::bind_copy_functions<Capsule>(capsule);
    capsule.def(py::init<float, float>(), "Create a Capsule", "radius"_a,
                "height"_a)
            .def(py::init<float, float, const Eigen::Matrix4f &>(),
                 "Create a Capsule", "radius"_a, "height"_a, "transform"_a)
            .def_readwrite("radius", &Capsule::radius_)
            .def_readwrite("height", &Capsule::height_);

    py::class_<PrimitivePack, std::shared_ptr<PrimitivePack>> pack(
            m, "PrimitivePack", "Packed primitive class.");
    py::detail::bind_default_constructor<PrimitivePack>(pack);
    py::detail::bind_copy_functions<PrimitivePack>(pack);
    pack.def_readwrite("box", &PrimitivePack::box_)
            .def_readwrite("sphere", &PrimitivePack::sphere_)
            .def_readwrite("capsule", &PrimitivePack::capsule_);

    py::class_<wrapper::device_vector_primitives,
               std::shared_ptr<wrapper::device_vector_primitives>>
            primitive_array(m, "PrimitiveArray",
                            "Packed primitive array class.");
    py::detail::bind_default_constructor<PrimitivePack>(pack);
    py::detail::bind_copy_functions<PrimitivePack>(pack);
    primitive_array.def("append",
                        &wrapper::device_vector_primitives::push_back);

    m.def("create_voxel_grid", &CreateVoxelGrid);
    m.def("create_voxel_grid_with_sweeping", &CreateVoxelGridWithSweeping);
    m.def("create_triangle_mesh", &CreateTriangleMesh);
}