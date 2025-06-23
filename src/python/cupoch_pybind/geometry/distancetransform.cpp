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
#include "cupoch/geometry/distancetransform.h"

#include "cupoch/camera/pinhole_camera_parameters.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch_pybind/device_map_wrapper.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/eigen_type_caster.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_distancetransform(py::module &m) {
    py::class_<geometry::DistanceVoxel,
               std::shared_ptr<geometry::DistanceVoxel>>
            voxel(m, "DistanceVoxel",
                  "Distance Voxel class, containing nearest grid id and distance");
    py::detail::bind_default_constructor<geometry::DistanceVoxel>(voxel);
    py::detail::bind_copy_functions<geometry::DistanceVoxel>(voxel);
    voxel.def("__repr__",
              [](const geometry::DistanceVoxel &voxel) {
                  std::ostringstream repr;
                  repr << "geometry::DistanceVoxel with nearest_index: ("
                       << voxel.nearest_index_(0) << ", " << voxel.nearest_index_(1)
                       << ", " << voxel.nearest_index_(2)
                       << "), distance: " << voxel.distance_ << ")";
                  return repr.str();
              })
            .def_readwrite("nearest_index", &geometry::DistanceVoxel::nearest_index_,
                           "Int numpy array of shape (3,): Grid coordinate "
                           "index of the nearest voxel.")
            .def_readwrite("distance", &geometry::DistanceVoxel::distance_,
                           "Float32: Distance from the nearest voxel.");

    py::class_<geometry::DistanceTransform, PyGeometry3D<geometry::DistanceTransform>,
               std::shared_ptr<geometry::DistanceTransform>,
               geometry::GeometryBase3D>
            distancetransform(m, "DistanceTransform",
                              "Distance transform is a collection of voxels which is a "
                              "special voxel grid "
                              "with a parameter of distance from the nearest voxel.");
    py::detail::bind_default_constructor<geometry::DistanceTransform>(
            distancetransform);
    py::detail::bind_copy_functions<geometry::DistanceTransform>(distancetransform);
    distancetransform
            .def(py::init<float, int, const Eigen::Vector3f &>(),
                 "Create a Distance transform", "voxel_size"_a, "resolution"_a,
                 "origin"_a = Eigen::Vector3f::Zero())
            .def("__repr__",
                 [](const geometry::DistanceTransform &distancetransform) {
                     return std::string("geometry::DistanceTransform with ") +
                            std::to_string(distancetransform.resolution_) +
                            " resolution.";
                 })
            .def("reconstruct", &geometry::DistanceTransform::Reconstruct,
                 "Reconstruct distance transform.")
            .def("compute_edt",
                 py::overload_cast<const geometry::VoxelGrid&>(
                         &geometry::DistanceTransform::ComputeEDT),
                 "Function to compute EDT from voxel grid.")
            .def("get_distance", &geometry::DistanceTransform::GetDistance)
            .def("get_distances",
                 [] (const geometry::DistanceTransform& self,
                     const wrapper::device_vector_vector3f& points) -> wrapper::device_vector_float {
                     return wrapper::device_vector_float(*self.GetDistances(points.data_));
                 })
            .def_static(
                    "create_from_occupancy_grid",
                    &geometry::DistanceTransform::CreateFromOccupancyGrid,
                    "Function to make voxels from a Occupancy Grid")
            .def_readwrite("voxel_size", &geometry::DistanceTransform::voxel_size_)
            .def_readwrite("resolution", &geometry::DistanceTransform::resolution_)
            .def_readwrite("origin", &geometry::DistanceTransform::origin_);
}