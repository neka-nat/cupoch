#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/geometry3d.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/helper.h"
#include <thrust/gather.h>

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

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

PointCloud::PointCloud() : Geometry(Geometry::GeometryType::PointCloud, 3) {}
PointCloud::PointCloud(const thrust::host_vector<Eigen::Vector3f_u>& points) : Geometry(Geometry::GeometryType::PointCloud, 3), points_(points) {}
PointCloud::PointCloud(const PointCloud& other) : Geometry(Geometry::GeometryType::PointCloud, 3), points_(other.points_), normals_(other.normals_), colors_(other.colors_) {}

PointCloud::~PointCloud() {}

void PointCloud::SetPoints(const thrust::host_vector<Eigen::Vector3f_u>& points) {
    points_ = points;
}

thrust::host_vector<Eigen::Vector3f_u> PointCloud::GetPoints() const {
    thrust::host_vector<Eigen::Vector3f_u> points = points_;
    return points;
}

void PointCloud::SetNormals(const thrust::host_vector<Eigen::Vector3f_u>& normals) {
    normals_ = normals;
}

thrust::host_vector<Eigen::Vector3f_u> PointCloud::GetNormals() const {
    thrust::host_vector<Eigen::Vector3f_u> normals = normals_;
    return normals;
}

void PointCloud::SetColors(const thrust::host_vector<Eigen::Vector3f_u>& colors) {
    colors_ = colors;
}

thrust::host_vector<Eigen::Vector3f_u> PointCloud::GetColors() const {
    thrust::host_vector<Eigen::Vector3f_u> colors = colors_;
    return colors;
}

PointCloud &PointCloud::Clear() {
    points_.clear();
    normals_.clear();
    colors_.clear();
    return *this;
}

bool PointCloud::IsEmpty() const {return !HasPoints();}

Eigen::Vector3f PointCloud::GetMinBound() const {
    return ComputeMinBound(points_);
}

Eigen::Vector3f PointCloud::GetMaxBound() const {
    return ComputeMaxBound(points_);
}

Eigen::Vector3f PointCloud::GetCenter() const {
    return ComuteCenter(points_);
}

PointCloud &PointCloud::NormalizeNormals() {
    thrust::for_each(normals_.begin(), normals_.end(), [] __device__ (Eigen::Vector3f_u& nl) {nl.normalize();});
    return *this;
}

PointCloud& PointCloud::Transform(const Eigen::Matrix4f& transformation) {
    TransformPoints(transformation, points_);
    TransformNormals(transformation, normals_);
    return *this;
}

std::shared_ptr<PointCloud> PointCloud::Crop(const Eigen::Vector3f &min_bound,
                                             const Eigen::Vector3f &max_bound) const {
    auto output = std::make_shared<PointCloud>();
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
