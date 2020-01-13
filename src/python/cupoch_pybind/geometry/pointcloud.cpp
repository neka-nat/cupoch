#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"
#include "cupoch_pybind/dl_converter.h"
#include "cupoch_pybind/docstring.h"

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
            .def("__repr__",
                 [](const geometry::PointCloud &pcd) {
                     return std::string("geometry::PointCloud with ") +
                            std::to_string(pcd.points_.size()) + " points.";
                 })
            .def_property("points", &geometry::PointCloud::GetPoints,
                                    &geometry::PointCloud::SetPoints)
            .def_property("normals", &geometry::PointCloud::GetNormals,
                                     &geometry::PointCloud::SetNormals)
            .def_property("colors", &geometry::PointCloud::GetColors,
                                    &geometry::PointCloud::SetColors)
            .def("to_points_dlpack", [](geometry::PointCloud &pcd) {return dlpack::ToDLpackCapsule(pcd.points_);})
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
            .def("paint_uniform_color",
                 &geometry::PointCloud::PaintUniformColor, "color"_a,
                 "Assigns each point in the PointCloud the same color.")
            .def("select_down_sample", &geometry::PointCloud::SelectDownSample,
                 "Function to select points from input pointcloud into output "
                 "pointcloud. ``indices``: "
                 "Indices of points to be selected. ``invert``: Set to "
                 "``True`` to "
                 "invert the selection of indices.",
                 "indices"_a, "invert"_a = false)
            .def("voxel_down_sample", &geometry::PointCloud::VoxelDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud with "
                 "a voxel",
                 "voxel_size"_a)
            .def("uniform_down_sample",
                 &geometry::PointCloud::UniformDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud "
                 "uniformly. The sample is performed in the order of the "
                 "points with "
                 "the 0-th point always chosen, not at random.",
                 "every_k_points"_a)
            .def("remove_none_finite_points",
                 &geometry::PointCloud::RemoveNoneFinitePoints,
                 "Function to remove none-finite points from the PointCloud",
                 "remove_nan"_a = true, "remove_infinite"_a = true)
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
                 "eps"_a, "min_points"_a, "print_progress"_a = false)
            .def_static(
                    "create_from_depth_image",
                    &geometry::PointCloud::CreateFromDepthImage,
                    R"(Factory function to create a pointcloud from a depth image and a
        camera. Given depth value d at (u, v) image coordinate, the corresponding 3d
        point is:
              - z = d / depth_scale
              - x = (u - cx) * z / fx
              - y = (v - cy) * z / fy
        )",
                    "depth"_a, "intrinsic"_a,
                    "extrinsic"_a = Eigen::Matrix4f::Identity(),
                    "depth_scale"_a = 1000.0, "depth_trunc"_a = 1000.0,
                    "stride"_a = 1);
    docstring::ClassMethodDocInject(m, "PointCloud", "has_colors");
    docstring::ClassMethodDocInject(m, "PointCloud", "has_normals");
    docstring::ClassMethodDocInject(m, "PointCloud", "has_points");
    docstring::ClassMethodDocInject(m, "PointCloud", "normalize_normals");
    docstring::ClassMethodDocInject(
            m, "PointCloud", "paint_uniform_color",
            {{"color", "RGB color for the PointCloud."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "select_down_sample",
            {{"indices", "Indices of points to be selected."},
             {"invert",
              "Set to ``True`` to invert the selection of indices."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "voxel_down_sample",
            {{"voxel_size", "Voxel size to downsample into."},
             {"invert", "set to ``True`` to invert the selection of indices"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "uniform_down_sample",
            {{"every_k_points",
              "Sample rate, the selected point indices are [0, k, 2k, ...]"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "remove_none_finite_points",
            {{"remove_nan", "Remove NaN values from the PointCloud"},
             {"remove_infinite",
              "Remove infinite values from the PointCloud"}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "remove_radius_outlier",
            {{"nb_points", "Number of points within the radius."},
             {"radius", "Radius of the sphere."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "remove_statistical_outlier",
            {{"nb_neighbors", "Number of neighbors around the target point."},
             {"std_ratio", "Standard deviation ratio."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "estimate_normals",
            {{"search_param",
              "The KDTree search parameters for neighborhood search."},
             {"fast_normal_computation",
              "If true, the normal estiamtion uses a non-iterative method to "
              "extract the eigenvector from the covariance matrix. This is "
              "faster, but is not as numerical stable."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "orient_normals_to_align_with_direction",
            {{"orientation_reference",
              "Normals are oriented with respect to orientation_reference."}});
    docstring::ClassMethodDocInject(
            m, "PointCloud", "cluster_dbscan",
            {{"eps",
              "Density parameter that is used to find neighbouring points."},
             {"min_points", "Minimum number of points to form a cluster."},
             {"print_progress",
              "If true the progress is visualized in the console."}});
    docstring::ClassMethodDocInject(m, "PointCloud", "create_from_depth_image");
}