#include "cupoc/geometry/pointcloud.h"
#include "cupoc/geometry/geometry3d.h"
#include "cupoc/utility/console.h"
#include "cupoc/utility/helper.h"
#include <thrust/gather.h>

using namespace cupoc;
using namespace cupoc::geometry;

namespace {

struct transform_functor {
    transform_functor(const Eigen::Matrix4f& transform) : transform_(transform){};
    const Eigen::Matrix4f_u transform_;
    __device__
    void operator()(Eigen::Vector3f_u& pt) {
        pt = transform_.block<3, 3>(0, 0) * pt + transform_.block<3, 1>(0, 3);
    }
};

struct cropped_copy_functor {
    cropped_copy_functor(const Eigen::Vector3f &min_bound, const Eigen::Vector3f &max_bound)
        : min_bound_(min_bound), max_bound_(max_bound) {};
    const Eigen::Vector3f min_bound_;
    const Eigen::Vector3f max_bound_;
    __device__
    bool operator()(const Eigen::Vector3f_u& pt) {
        if (pt[0] >= min_bound_[0] && pt[0] <= max_bound_[0] &&
            pt[1] >= min_bound_[1] && pt[1] <= max_bound_[1] &&
            pt[2] >= min_bound_[2] && pt[2] <= max_bound_[2]) {
            return true;
        } else {
            return false;
        }
    }
};

}

PointCloud::PointCloud() {}

PointCloud::~PointCloud() {}

Eigen::Vector3f PointCloud::GetMinBound() const {
    return ComputeMinBound(points_);
}

Eigen::Vector3f PointCloud::GetMaxBound() const {
    return ComputeMaxBound(points_);
}

Eigen::Vector3f PointCloud::GetCenter() const {
    return ComuteCenter(points_);
}

PointCloud& PointCloud::Transform(const Eigen::Matrix4f& transformation) {
    transform_functor func(transformation);
    thrust::for_each(points_.begin(), points_.end(), func);
    return *this;
}

utility::shared_ptr<PointCloud> PointCloud::Crop(const Eigen::Vector3f &min_bound,
                                                 const Eigen::Vector3f &max_bound) const {
    auto output = utility::shared_ptr<PointCloud>(new PointCloud());
    if (min_bound[0] > max_bound[0] ||
        min_bound[1] > max_bound[1] ||
        min_bound[2] > max_bound[2]) {
        utility::LogWarning(
                "[CropPointCloud] Illegal boundary clipped all points.\n");
        return output;
    }

    output->points_.resize(points_.size());
    cropped_copy_functor func(min_bound, max_bound);
    thrust::device_vector<Eigen::Vector3f_u>::iterator end = thrust::copy_if(points_.begin(), points_.end(), output->points_.begin(), func);
    output->points_.resize(static_cast<int>(end - output->points_.begin()));
    return output;
}
