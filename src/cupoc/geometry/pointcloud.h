#pragma once
#include "cupoc/utility/shared_ptr.hpp"
#include "cupoc/utility/eigen.h"
#include "cupoc/geometry/kdtree_search_param.h"
#include <thrust/device_vector.h>

namespace cupoc {
namespace geometry {

class PointCloud {
public:
    PointCloud();
    ~PointCloud();

    Eigen::Vector3f GetMinBound() const;
    Eigen::Vector3f GetMaxBound() const;
    Eigen::Vector3f GetCenter() const;

    __host__ __device__
    bool HasPoints() const { return !points_.empty(); }

    __host__ __device__
    bool HasNormals() const {
        return !points_.empty() && normals_.size() == points_.size();
    }

    __host__ __device__
    bool HasColors() const {
        return !points_.empty() && colors_.size() == points_.size();
    }

    PointCloud& Transform(const Eigen::Matrix4f& transformation);

    utility::shared_ptr<PointCloud> SelectDownSample(const thrust::device_vector<size_t> &indices, bool invert = false) const;

    utility::shared_ptr<PointCloud> VoxelDownSample(float voxel_size) const;

    utility::shared_ptr<PointCloud> Crop(const Eigen::Vector3f &min_bound,
                                         const Eigen::Vector3f &max_bound) const;

    bool EstimateNormals(const KDTreeSearchParam& search_param = KDTreeSearchParamKNN());
    bool OrientNormalsToAlignWithDirection(const Eigen::Vector3f &orientation_reference = Eigen::Vector3f(0.0, 0.0, 1.0));

public:
    thrust::device_vector<Eigen::Vector3f_u> points_;
    thrust::device_vector<Eigen::Vector3f_u> normals_;
    thrust::device_vector<Eigen::Vector3f_u> colors_;
};


}
}