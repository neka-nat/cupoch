#pragma once
#include <thrust/host_vector.h>
#include "cupoc/utility/eigen.h"
#include "cupoc/geometry/pointcloud.h"

namespace cupoc {
namespace geometry {

class host_PointCloud {
public:
    host_PointCloud();
    ~host_PointCloud();
    host_PointCloud(const thrust::host_vector<Eigen::Vector3f_u>& points);
    host_PointCloud(const host_PointCloud& other);

    void SetPoints(const thrust::host_vector<Eigen::Vector3f_u>& points);
    thrust::host_vector<Eigen::Vector3f_u> GetPoints() const;

    bool HasPoints() const;
    bool HasNormals() const;
    bool HasColors() const;

    host_PointCloud& Transform(const Eigen::Matrix4f& transformation);

    host_PointCloud VoxelDownSample(float voxel_size);

private:
    utility::shared_ptr<PointCloud> device_impl_;
};

}
}