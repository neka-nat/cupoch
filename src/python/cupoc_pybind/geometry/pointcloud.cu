#include "cupoc_pybind/geometry/pointcloud.h"

using namespace cupoc;
using namespace cupoc::geometry;

host_PointCloud::host_PointCloud() : device_impl_(new PointCloud()) {}
host_PointCloud::~host_PointCloud() {}
host_PointCloud::host_PointCloud(const host_PointCloud& other) {*device_impl_ = *(other.device_impl_);}
host_PointCloud::host_PointCloud(const thrust::host_vector<Eigen::Vector3f_u>& points) {device_impl_->points_ = points;}

void host_PointCloud::SetPoints(const thrust::host_vector<Eigen::Vector3f_u>& points) {
    device_impl_->points_ = points;
}

thrust::host_vector<Eigen::Vector3f_u> host_PointCloud::GetPoints() const {
    thrust::host_vector<Eigen::Vector3f_u> points = device_impl_->points_;
    return std::move(points);
}

bool host_PointCloud::HasPoints() const {return device_impl_->HasPoints();}
bool host_PointCloud::HasNormals() const {return device_impl_->HasNormals();}
bool host_PointCloud::HasColors() const {return device_impl_->HasColors();}

host_PointCloud& host_PointCloud::Transform(const Eigen::Matrix4f& transformation) {
    device_impl_->Transform(transformation);
    return *this;   
}

host_PointCloud host_PointCloud::VoxelDownSample(float voxel_size) {
    host_PointCloud out;
    out.device_impl_ = device_impl_->VoxelDownSample(voxel_size);
    return out;
}
