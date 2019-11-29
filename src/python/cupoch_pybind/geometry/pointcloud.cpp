#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch/geometry/pointcloud.h"

using namespace cupoch;

void pybind_pointcloud(py::module &m) {
    py::class_<geometry::PointCloud, std::shared_ptr<geometry::PointCloud>>
            pointcloud(m, "PointCloud",
                       "PointCloud class. A point cloud consists of point "
                       "coordinates, and optionally point colors and point "
                       "normals.");
    py::detail::bind_default_constructor<geometry::PointCloud>(pointcloud);
    py::detail::bind_copy_functions<geometry::PointCloud>(pointcloud);
    pointcloud
            .def(py::init<const thrust::host_vector<Eigen::Vector3f_u> &>(),
                 "Create a PointCloud from points", "points"_a)
            .def_property("points", &geometry::PointCloud::GetPoints,
                                    &geometry::PointCloud::SetPoints)
            .def_property("normals", &geometry::PointCloud::GetNormals,
                                     &geometry::PointCloud::SetNormals)
            .def_property("colors", &geometry::PointCloud::GetColors,
                                     &geometry::PointCloud::SetColors)
            .def("has_points", &geometry::PointCloud::HasPoints,
                 "Returns ``True`` if the point cloud contains points.")
            .def("has_normals", &geometry::PointCloud::HasNormals,
                 "Returns ``True`` if the point cloud contains point normals.")
            .def("has_colors", &geometry::PointCloud::HasColors,
                 "Returns ``True`` if the point cloud contains point colors.")
            .def("normalize_normals", &geometry::PointCloud::NormalizeNormals,
                 "Normalize point normals to length 1.")
            .def("transform", &geometry::PointCloud::Transform,
                 "Apply transformation (4x4 matrix) to the geometry "
                 "coordinates.")
            .def("voxel_down_sample", &geometry::PointCloud::VoxelDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud with "
                 "a voxel",
                 "voxel_size"_a)
            .def("estimate_normals", &geometry::PointCloud::EstimateNormals,
                 "Function to compute the normals of a point cloud. Normals "
                 "are oriented with respect to the input point cloud if "
                 "normals exist",
                 "search_param"_a = geometry::KDTreeSearchParamKNN())
            .def("orient_normals_to_align_with_direction",
                 &geometry::PointCloud::OrientNormalsToAlignWithDirection,
                 "Function to orient the normals of a point cloud",
                 "orientation_reference"_a = Eigen::Vector3f(0.0, 0.0, 1.0));
}