#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/camera/pinhole_camera_parameters.h"
#include "cupoch/geometry/image.h"
#include "cupoch/geometry/pointcloud.h"

#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/device_map_wrapper.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_occupanygrid(py::module &m) {
    py::class_<wrapper::OccupancyVoxelMap, std::shared_ptr<wrapper::OccupancyVoxelMap>> voxel_map(m, "DeviceOccupancyVoxelMap");
    voxel_map.def(py::init<>())
             .def("__len__", &wrapper::OccupancyVoxelMap::size)
             .def("cpu", &wrapper::OccupancyVoxelMap::cpu);
    py::class_<geometry::OccupancyVoxel, std::shared_ptr<geometry::OccupancyVoxel>> voxel(
            m, "OccupancyVoxel", "Occupancy Voxel class, containing grid id, occupancy log odds and color");
    py::detail::bind_default_constructor<geometry::OccupancyVoxel>(voxel);
    py::detail::bind_copy_functions<geometry::OccupancyVoxel>(voxel);
    voxel.def("__repr__",
              [](const geometry::OccupancyVoxel &voxel) {
                  std::ostringstream repr;
                  repr << "geometry::OccupancyVoxel with grid_index: ("
                       << voxel.grid_index_(0) << ", " << voxel.grid_index_(1)
                       << ", " << voxel.grid_index_(2) << "), prob_log: "
                       << voxel.prob_log_ << ", color: ("
                       << voxel.color_(0) << ", " << voxel.color_(1) << ", "
                       << voxel.color_(2) << ")";
                  return repr.str();
              })
            .def(py::init([](const Eigen::Vector3i &grid_index) {
                     return new geometry::OccupancyVoxel(grid_index);
                 }),
                 "grid_index"_a)
            .def(py::init([](const Eigen::Vector3i &grid_index, float prob_log) {
                     return new geometry::OccupancyVoxel(grid_index, prob_log);
                 }),
                 "grid_index"_a, "prob_log"_a)
            .def(py::init([](const Eigen::Vector3i &grid_index,
                             float prob_log,
                             const Eigen::Vector3f &color) {
                     return new geometry::OccupancyVoxel(grid_index, prob_log, color);
                 }),
                 "grid_index"_a, "prob_log"_a, "color"_a)
            .def_readwrite("grid_index", &geometry::OccupancyVoxel::grid_index_,
                           "Int numpy array of shape (3,): Grid coordinate "
                           "index of the voxel.")
            .def_readwrite(
                    "prob_log", &geometry::OccupancyVoxel::prob_log_,
                    "Float32: Log odds of the voxel.")
            .def_readwrite(
                    "color", &geometry::OccupancyVoxel::color_,
                    "Float32 numpy array of shape (3,): Color of the voxel.");

}