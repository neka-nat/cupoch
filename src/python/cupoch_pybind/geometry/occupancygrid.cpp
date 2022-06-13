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
#include "cupoch/geometry/occupancygrid.h"

#include "cupoch/camera/pinhole_camera_parameters.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch_pybind/device_map_wrapper.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/eigen_type_caster.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_occupanygrid(py::module &m) {
    py::class_<geometry::OccupancyVoxel,
               std::shared_ptr<geometry::OccupancyVoxel>>
            voxel(m, "OccupancyVoxel",
                  "Occupancy Voxel class, containing grid id, occupancy log "
                  "odds and color");
    py::detail::bind_default_constructor<geometry::OccupancyVoxel>(voxel);
    py::detail::bind_copy_functions<geometry::OccupancyVoxel>(voxel);
    voxel.def("__repr__",
              [](const geometry::OccupancyVoxel &voxel) {
                  std::ostringstream repr;
                  repr << "geometry::OccupancyVoxel with grid_index: ("
                       << voxel.grid_index_(0) << ", " << voxel.grid_index_(1)
                       << ", " << voxel.grid_index_(2)
                       << "), prob_log: " << voxel.prob_log_ << ", color: ("
                       << voxel.color_(0) << ", " << voxel.color_(1) << ", "
                       << voxel.color_(2) << ")";
                  return repr.str();
              })
            .def(py::init([](const Eigen::Vector3i &grid_index) {
                     return new geometry::OccupancyVoxel(grid_index);
                 }),
                 "grid_index"_a)
            .def(py::init([](const Eigen::Vector3i &grid_index,
                             float prob_log) {
                     return new geometry::OccupancyVoxel(grid_index, prob_log);
                 }),
                 "grid_index"_a, "prob_log"_a)
            .def(py::init([](const Eigen::Vector3i &grid_index, float prob_log,
                             const Eigen::Vector3f &color) {
                     return new geometry::OccupancyVoxel(grid_index, prob_log,
                                                         color);
                 }),
                 "grid_index"_a, "prob_log"_a, "color"_a)
            .def_readwrite("grid_index", &geometry::OccupancyVoxel::grid_index_,
                           "Int numpy array of shape (3,): Grid coordinate "
                           "index of the voxel.")
            .def_readwrite("prob_log", &geometry::OccupancyVoxel::prob_log_,
                           "Float32: Log odds of the voxel.")
            .def_readwrite(
                    "color", &geometry::OccupancyVoxel::color_,
                    "Float32 numpy array of shape (3,): Color of the voxel.");

    py::class_<geometry::OccupancyGrid, PyGeometry3D<geometry::OccupancyGrid>,
               std::shared_ptr<geometry::OccupancyGrid>,
               geometry::GeometryBase3D>
            occupancygrid(m, "OccupancyGrid",
                          "Occupancy is a collection of voxels which is a "
                          "special voxel grid "
                          "with a parameter of occupancy probability.");
    py::detail::bind_default_constructor<geometry::OccupancyGrid>(
            occupancygrid);
    py::detail::bind_copy_functions<geometry::OccupancyGrid>(occupancygrid);
    occupancygrid
            .def(py::init<float, int, const Eigen::Vector3f &>(),
                 "Create a Occupancy grid", "voxel_size"_a, "resolution"_a,
                 "origin"_a = Eigen::Vector3f::Zero())
            .def("__repr__",
                 [](const geometry::OccupancyGrid &occupancygrid) {
                     return std::string("geometry::OccupancyGrid with ") +
                            std::to_string(occupancygrid.ExtractKnownVoxels()
                                                   ->size()) +
                            " voxels.";
                 })
            .def_property_readonly("voxels",
                                   [](const geometry::OccupancyGrid &og) {
                                       return wrapper::device_vector_occupancyvoxel(*og.ExtractKnownVoxels());
                                   })
            .def("reconstruct", &geometry::OccupancyGrid::Reconstruct,
                 "Reconstruct dense voxel grid.")
            .def("insert",
                 py::overload_cast<const geometry::PointCloud &,
                                   const Eigen::Vector3f &, float>(
                         &geometry::OccupancyGrid::Insert),
                 "Function to insert occupancy grid from pointcloud.",
                 "pointcloud"_a, "viewpoint"_a, "max_range"_a = -1.0)
            .def("set_free_area", &geometry::OccupancyGrid::SetFreeArea)
            .def_static(
                    "create_from_voxel_grid",
                    &geometry::OccupancyGrid::CreateFromVoxelGrid,
                    "Function to make occupancy grid from a Voxel Grid")
            .def_readwrite("voxel_size", &geometry::OccupancyGrid::voxel_size_)
            .def_readwrite("resolution", &geometry::OccupancyGrid::resolution_)
            .def_readwrite("origin", &geometry::OccupancyGrid::origin_)
            .def_readwrite("clamping_thres_min",
                           &geometry::OccupancyGrid::clamping_thres_min_)
            .def_readwrite("clamping_thres_max",
                           &geometry::OccupancyGrid::clamping_thres_max_)
            .def_readwrite("prob_hit_log",
                           &geometry::OccupancyGrid::prob_hit_log_)
            .def_readwrite("prob_miss_log",
                           &geometry::OccupancyGrid::prob_miss_log_)
            .def_readwrite("occ_prob_thres_log",
                           &geometry::OccupancyGrid::occ_prob_thres_log_)
            .def_readwrite("visualize_free_area",
                           &geometry::OccupancyGrid::visualize_free_area_);
}