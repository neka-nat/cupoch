#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/camera/pinhole_camera_parameters.h"
#include "cupoch/geometry/pointcloud.h"

#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/device_map_wrapper.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"
#include "cupoch_pybind/geometry/eigen_type_caster.h"

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

    py::class_<geometry::OccupancyGrid, PyGeometry3D<geometry::OccupancyGrid>,
               std::shared_ptr<geometry::OccupancyGrid>, geometry::Geometry3D>
            occupancygrid(m, "OccupancyGrid",
                          "Occupancy is a collection of voxels which is a special voxel grid "
                          "with a parameter of occupancy probability.");
    py::detail::bind_default_constructor<geometry::OccupancyGrid>(occupancygrid);
    py::detail::bind_copy_functions<geometry::OccupancyGrid>(occupancygrid);
    occupancygrid
            .def(py::init<float, const Eigen::Vector3f&>(),
                 "Create a Occupancy grid", "voxel_grid"_a, "origin"_a = Eigen::Vector3f::Zero())
            .def("__repr__",
                 [](const geometry::OccupancyGrid &occupancygrid) {
                     return std::string("geometry::OccupancyGrid with ") +
                            std::to_string(occupancygrid.voxels_keys_.size()) +
                            " voxels.";
                 })
            .def_property("voxels", [] (geometry::OccupancyGrid &og) {return wrapper::OccupancyVoxelMap(og.voxels_keys_, og.voxels_values_);},
                                    [] (geometry::OccupancyGrid &og, const wrapper::OccupancyVoxelMap& map) {
                                        wrapper::FromWrapper(og.voxels_keys_, og.voxels_values_, map);})
            .def("insert", py::overload_cast<const geometry::PointCloud&, const Eigen::Vector3f&, float>(&geometry::OccupancyGrid::Insert),
                 "Function to insert occupancy grid from pointcloud.",
                 "pointcloud"_a, "viewpoint"_a, "max_range"_a = -1.0)
            .def_readwrite("clamping_thres_min", &geometry::OccupancyGrid::clamping_thres_min_)
            .def_readwrite("clamping_thres_max", &geometry::OccupancyGrid::clamping_thres_max_)
            .def_readwrite("prob_hit_log", &geometry::OccupancyGrid::prob_hit_log_)
            .def_readwrite("prob_miss_log", &geometry::OccupancyGrid::prob_miss_log_)
            .def_readwrite("occ_prob_thres_log", &geometry::OccupancyGrid::occ_prob_thres_log_)
            .def_readwrite("visualize_free_area", &geometry::OccupancyGrid::visualize_free_area_);
}