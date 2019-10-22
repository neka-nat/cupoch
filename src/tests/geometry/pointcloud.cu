#include "cupoc/geometry/pointcloud.h"
#include "tests/geometry/pointcloud.h"
#include <thrust/host_vector.h>

using namespace cupoc;
using namespace cupoc::geometry;

#define DEG2RAD(deg) (deg * M_PI / 180.0)

void cupoc::geometry::test_pointcloud_transform() {
    thrust::host_vector<Eigen::Vector3f_u> points;
    for (size_t i = 0; i < 1000; ++i) points.push_back(Eigen::Vector3f(1.0, 0.0, 0.0));
    PointCloud cloud;
    cloud.points_.resize(points.size());
    thrust::copy(points.begin(), points.end(), cloud.points_.begin());
    Eigen::Matrix4f trans = (Eigen::Matrix4f() << std::cos(DEG2RAD(30.0)), -std::sin(DEG2RAD(30.0)), 0.0, 0.0,
                                                  std::sin(DEG2RAD(30.0)), std::cos(DEG2RAD(30.0)), 0.0, 0.0,
                                                  0.0, 0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0, 1.0).finished();
    cloud.Transform(trans);
    points = cloud.points_;
    std::cout << points[0] << std::endl;
}

void cupoc::geometry::test_pointcloud_voxel_down_sampling() {
    thrust::host_vector<Eigen::Vector3f_u> points;
    for (size_t i = 0; i < 1000; ++i) points.push_back(Eigen::Vector3f(1.0, 0.0, 0.0));
    PointCloud cloud;
    cloud.points_.resize(points.size());
    thrust::copy(points.begin(), points.end(), cloud.points_.begin());
    auto out = cloud.VoxelDownSample(0.01);
}