#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_pointcloud(py::module &m) {
    py::class_<geometry::PointCloud, PyGeometry3D<geometry::PointCloud>,
               std::shared_ptr<geometry::PointCloud>, geometry::Geometry3D>
            pointcloud(m, "PointCloud",
                       "PointCloud class. A point cloud consists of point "
                       "coordinates, and optionally point colors and point "
                       "normals.");
    py::detail::bind_default_constructor<geometry::PointCloud>(pointcloud);
    py::detail::bind_copy_functions<geometry::PointCloud>(pointcloud);
    pointcloud
            .def(py::init<const thrust::host_vector<Eigen::Vector3f> &>(),
                 "Create a PointCloud from points", "points"_a)
            .def_property("host_points", &geometry::PointCloud::GetPoints,
                                         &geometry::PointCloud::SetPoints)
            .def_property("host_normals", &geometry::PointCloud::GetNormals,
                                          &geometry::PointCloud::SetNormals)
            .def_property("host_colors", &geometry::PointCloud::GetColors,
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
            .def("remove_radius_outlier",
                 &geometry::PointCloud::RemoveRadiusOutliersHost,
                 "Function to remove points that have less than nb_points"
                 " in a given sphere of a given radius",
                 "nb_points"_a, "radius"_a)
            .def("remove_statistical_outlier",
                 &geometry::PointCloud::RemoveStatisticalOutliersHost,
                 "Function to remove points that are further away from their "
                 "neighbors in average",
                 "nb_neighbors"_a, "std_ratio"_a)
            .def("estimate_normals", &geometry::PointCloud::EstimateNormals,
                 "Function to compute the normals of a point cloud. Normals "
                 "are oriented with respect to the input point cloud if "
                 "normals exist",
                 "search_param"_a = geometry::KDTreeSearchParamKNN())
            .def("orient_normals_to_align_with_direction",
                 &geometry::PointCloud::OrientNormalsToAlignWithDirection,
                 "Function to orient the normals of a point cloud",
                 "orientation_reference"_a = Eigen::Vector3f(0.0, 0.0, 1.0))
            .def("cluster_dbscan", &geometry::PointCloud::ClusterDBSCANHost,
                 "Cluster PointCloud using the DBSCAN algorithm  Ester et al., "
                 "'A Density-Based Algorithm for Discovering Clusters in Large "
                 "Spatial Databases with Noise', 1996. Returns a list of point "
                 "labels, -1 indicates noise according to the algorithm.",
                 "eps"_a, "min_points"_a, "print_progress"_a = false);
}