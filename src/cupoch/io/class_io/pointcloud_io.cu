#include "cupoch/io/class_io/pointcloud_io.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/helper.h"

using namespace cupoch;
using namespace cupoch::io;


void HostPointCloud::FromDevice(const geometry::PointCloud& pointcloud) {
    points_.resize(pointcloud.points_.size());
    normals_.resize(pointcloud.normals_.size());
    colors_.resize(pointcloud.colors_.size());
    utility::CopyFromDeviceMultiStream(pointcloud.points_, points_);
    utility::CopyFromDeviceMultiStream(pointcloud.normals_, normals_);
    utility::CopyFromDeviceMultiStream(pointcloud.colors_, colors_);
    cudaDeviceSynchronize();
}

void HostPointCloud::ToDevice(geometry::PointCloud& pointcloud) const {
    pointcloud.points_.resize(points_.size());
    pointcloud.normals_.resize(normals_.size());
    pointcloud.colors_.resize(colors_.size());
    utility::CopyToDeviceMultiStream(points_, pointcloud.points_);
    utility::CopyToDeviceMultiStream(normals_, pointcloud.normals_);
    utility::CopyToDeviceMultiStream(colors_, pointcloud.colors_);
    cudaDeviceSynchronize();
}

void HostPointCloud::Clear() {
    points_.clear();
    normals_.clear();
    colors_.clear();
}
