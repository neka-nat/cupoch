#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/camera/pinhole_camera_parameters.h"
#include "cupoch/geometry/image.h"
#include "cupoch/geometry/pointcloud.h"

#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

#include <sstream>

using namespace cupoch;

void pybind_voxelgrid(py::module &m) {
    py::class_<geometry::Voxel, std::shared_ptr<geometry::Voxel>> voxel(
            m, "Voxel", "Base Voxel class, containing grid id and color");
    py::detail::bind_default_constructor<geometry::Voxel>(voxel);
    py::detail::bind_copy_functions<geometry::Voxel>(voxel);
    voxel.def("__repr__",
              [](const geometry::Voxel &voxel) {
                  std::ostringstream repr;
                  repr << "geometry::Voxel with grid_index: ("
                       << voxel.grid_index_(0) << ", " << voxel.grid_index_(1)
                       << ", " << voxel.grid_index_(2) << "), color: ("
                       << voxel.color_(0) << ", " << voxel.color_(1) << ", "
                       << voxel.color_(2) << ")";
                  return repr.str();
              })
            .def(py::init([](const Eigen::Vector3i &grid_index) {
                     return new geometry::Voxel(grid_index);
                 }),
                 "grid_index"_a)
            .def(py::init([](const Eigen::Vector3i &grid_index,
                             const Eigen::Vector3f &color) {
                     return new geometry::Voxel(grid_index, color);
                 }),
                 "grid_index"_a, "color"_a)
            .def_readwrite("grid_index", &geometry::Voxel::grid_index_,
                           "Int numpy array of shape (3,): Grid coordinate "
                           "index of the voxel.")
            .def_readwrite(
                    "color", &geometry::Voxel::color_,
                    "Float32 numpy array of shape (3,): Color of the voxel.");

    py::class_<geometry::VoxelGrid, PyGeometry3D<geometry::VoxelGrid>,
               std::shared_ptr<geometry::VoxelGrid>, geometry::Geometry3D>
            voxelgrid(m, "VoxelGrid",
                      "VoxelGrid is a collection of voxels which are aligned "
                      "in grid.");
    py::detail::bind_default_constructor<geometry::VoxelGrid>(voxelgrid);
    py::detail::bind_copy_functions<geometry::VoxelGrid>(voxelgrid);
    voxelgrid
            .def("__repr__",
                 [](const geometry::VoxelGrid &voxelgrid) {
                     return std::string("geometry::VoxelGrid with ") +
                            std::to_string(voxelgrid.voxels_keys_.size()) +
                            " voxels.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_colors", &geometry::VoxelGrid::HasColors,
                 "Returns ``True`` if the voxel grid contains voxel colors.")
            .def("has_voxels", &geometry::VoxelGrid::HasVoxels,
                 "Returns ``True`` if the voxel grid contains voxels.")
            .def("get_voxel", &geometry::VoxelGrid::GetVoxel, "point"_a,
                 "Returns voxel index given query point.")
            .def("check_if_included", &geometry::VoxelGrid::CheckIfIncluded,
                 "queries"_a,
                 "Element-wise check if a query in the list is included in "
                 "the VoxelGrid. Queries are double precision and "
                 "are mapped to the closest voxel.")
            .def("carve_depth_map", &geometry::VoxelGrid::CarveDepthMap,
                 "depth_map"_a, "camera_params"_a,
                 "keep_voxels_outside_image"_a = false,
                 "Remove all voxels from the VoxelGrid where none of the "
                 "boundary points of the voxel projects to depth value that is "
                 "smaller, or equal than the projected depth of the boundary "
                 "point. If keep_voxels_outside_image is true then voxels are "
                 "only carved if all boundary points project to a valid image "
                 "location.")
            .def("carve_silhouette", &geometry::VoxelGrid::CarveSilhouette,
                 "silhouette_mask"_a, "camera_params"_a,
                 "keep_voxels_outside_image"_a = false,
                 "Remove all voxels from the VoxelGrid where none of the "
                 "boundary points of the voxel projects to a valid mask pixel "
                 "(pixel value > 0). If keep_voxels_outside_image is true then "
                 "voxels are only carved if all boundary points project to a "
                 "valid image location.")
            .def_static("create_dense", &geometry::VoxelGrid::CreateDense,
                        "Creates a voxel grid where every voxel is set (hence "
                        "dense). This is a useful starting point for voxel "
                        "carving",
                        "origin"_a, "voxel_size"_a, "width"_a, "height"_a,
                        "depth"_a)
            .def_static("create_from_point_cloud",
                        &geometry::VoxelGrid::CreateFromPointCloud,
                        "Function to make voxels from a PointCloud", "input"_a,
                        "voxel_size"_a)
            .def_static("create_from_point_cloud_within_bounds",
                        &geometry::VoxelGrid::CreateFromPointCloudWithinBounds,
                        "Function to make voxels from a PointCloud", "input"_a,
                        "voxel_size"_a, "min_bound"_a, "max_bound"_a)
            .def_static("create_from_triangle_mesh",
                        &geometry::VoxelGrid::CreateFromTriangleMesh,
                        "Function to make voxels from a TriangleMesh",
                        "input"_a, "voxel_size"_a)
            .def_static(
                    "create_from_triangle_mesh_within_bounds",
                    &geometry::VoxelGrid::CreateFromTriangleMeshWithinBounds,
                    "Function to make voxels from a PointCloud", "input"_a,
                    "voxel_size"_a, "min_bound"_a, "max_bound"_a)
            .def_readwrite("origin", &geometry::VoxelGrid::origin_,
                           "``float64`` vector of length 3: Coorindate of the "
                           "origin point.")
            .def_readwrite("voxel_size", &geometry::VoxelGrid::voxel_size_);
    docstring::ClassMethodDocInject(m, "VoxelGrid", "has_colors");
    docstring::ClassMethodDocInject(m, "VoxelGrid", "has_voxels");
    docstring::ClassMethodDocInject(m, "VoxelGrid", "get_voxel",
                                    {{"point", "The query point."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "check_if_included",
            {{"query", "a list of voxel indices to check."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "carve_depth_map",
            {{"depth_map", "Depth map (Image) used for VoxelGrid carving."},
             {"camera_parameters",
              "PinholeCameraParameters used to record the given depth_map."},
             {"keep_voxels_outside_image",
              "retain voxels that don't project"
              " to pixels in the image"}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "carve_silhouette",
            {{"silhouette_mask",
              "Silhouette mask (Image) used for VoxelGrid carving."},
             {"camera_parameters",
              "PinholeCameraParameters used to record the given depth_map."},
             {"keep_voxels_outside_image",
              "retain voxels that don't project"
              " to pixels in the image"}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "create_dense",
            {{"origin", "Coordinate center of the VoxelGrid"},
             {"voxel_size", "Voxel size of of the VoxelGrid construction."},
             {"width", "Spatial width extend of the VoxelGrid."},
             {"height", "Spatial height extend of the VoxelGrid."},
             {"depth", "Spatial depth extend of the VoxelGrid."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "create_from_point_cloud",
            {{"input", "The input PointCloud"},
             {"voxel_size", "Voxel size of of the VoxelGrid construction."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "create_from_point_cloud_within_bounds",
            {{"input", "The input PointCloud"},
             {"voxel_size", "Voxel size of of the VoxelGrid construction."},
             {"min_bound",
              "Minimum boundary point for the VoxelGrid to create."},
             {"max_bound",
              "Maximum boundary point for the VoxelGrid to create."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "create_from_triangle_mesh",
            {{"input", "The input TriangleMesh"},
             {"voxel_size", "Voxel size of of the VoxelGrid construction."}});
    docstring::ClassMethodDocInject(
            m, "VoxelGrid", "create_from_triangle_mesh_within_bounds",
            {{"input", "The input TriangleMesh"},
             {"voxel_size", "Voxel size of of the VoxelGrid construction."},
             {"min_bound",
              "Minimum boundary point for the VoxelGrid to create."},
             {"max_bound",
              "Maximum boundary point for the VoxelGrid to create."}});
}

void pybind_voxelgrid_methods(py::module &m) {}