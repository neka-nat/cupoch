#include "cupoc_pybind/geometry/pointcloud.h"
#include "cupoc_pybind/geometry/geometry.h"

using namespace cupoc;

void pybind_pointcloud(py::module &m) {
    py::class_<geometry::host_PointCloud>
            pointcloud(m, "PointCloud",
                       "PointCloud class. A point cloud consists of point "
                       "coordinates, and optionally point colors and point "
                       "normals.");
    py::detail::bind_default_constructor<geometry::host_PointCloud>(pointcloud);
    py::detail::bind_copy_functions<geometry::host_PointCloud>(pointcloud);
    pointcloud
            .def(py::init<const thrust::host_vector<Eigen::Vector3f_u> &>(),
                 "Create a PointCloud from points", "points"_a)
            .def_property("points", &geometry::host_PointCloud::GetPoints,
                                    &geometry::host_PointCloud::SetPoints)
            .def("has_points", &geometry::host_PointCloud::HasPoints,
                 "Returns ``True`` if the point cloud contains points.")
            .def("has_normals", &geometry::host_PointCloud::HasNormals,
                 "Returns ``True`` if the point cloud contains point normals.")
            .def("has_colors", &geometry::host_PointCloud::HasColors,
                 "Returns ``True`` if the point cloud contains point colors.")
            .def("transform", &geometry::host_PointCloud::Transform,
                 "Apply transformation (4x4 matrix) to the geometry "
                 "coordinates.")
            .def("voxel_down_sample", &geometry::host_PointCloud::VoxelDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud with "
                 "a voxel",
                 "voxel_size"_a);

}