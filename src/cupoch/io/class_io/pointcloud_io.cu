#include "cupoch/geometry/pointcloud.h"
#include "cupoch/io/class_io/pointcloud_io.h"
#include "cupoch/utility/helper.h"

using namespace cupoch;
using namespace cupoch::io;

void HostPointCloud::FromDevice(const geometry::PointCloud& pointcloud) {
    points_.resize(pointcloud.points_.size());
    normals_.resize(pointcloud.normals_.size());
    colors_.resize(pointcloud.colors_.size());
    thrust::copy(pointcloud.points_.begin(), pointcloud.points_.end(),
                 points_.begin());
    thrust::copy(pointcloud.normals_.begin(), pointcloud.normals_.end(),
                 normals_.begin());
    thrust::copy(pointcloud.colors_.begin(), pointcloud.colors_.end(),
                 colors_.begin());
}

void HostPointCloud::ToDevice(geometry::PointCloud& pointcloud) const {
    pointcloud.points_.resize(points_.size());
    pointcloud.normals_.resize(normals_.size());
    pointcloud.colors_.resize(colors_.size());
    thrust::copy(points_.begin(), points_.end(), pointcloud.points_.begin());
    thrust::copy(normals_.begin(), normals_.end(), pointcloud.normals_.begin());
    thrust::copy(colors_.begin(), colors_.end(), pointcloud.colors_.begin());
}

void HostPointCloud::Clear() {
    points_.clear();
    normals_.clear();
    colors_.clear();
}
