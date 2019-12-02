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

struct check_nan_functor {
    check_nan_functor(bool remove_nan, bool remove_infinite)
        : remove_nan_(remove_nan), remove_infinite_(remove_infinite) {};
    const bool remove_nan_;
    const bool remove_infinite_;
    __device__
    bool operator()(const Eigen::Vector3f_u& point) const {
        bool is_nan = remove_nan_ &&
                      (std::isnan(point(0)) || std::isnan(point(1)) ||
                       std::isnan(point(2)));
        bool is_infinite = remove_infinite_ && (std::isinf(point(0)) ||
                                                std::isinf(point(1)) ||
                                                std::isinf(point(2)));
        return is_nan || is_infinite;
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

PointCloud &PointCloud::PaintUniformColor(const Eigen::Vector3f &color) {
    ResizeAndPaintUniformColor(colors_, points_.size(), color);
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

    bool has_normal = HasNormals();
    bool has_color = HasColors();
    output->points_.resize(points_.size());
    cropped_copy_functor func(min_bound, max_bound);
    size_t n_out = 0;
    if (!has_normal && !has_color) {
        auto end = thrust::copy_if(points_.begin(), points_.end(), output->points_.begin(), func);
        n_out = thrust::distance(output->points_.begin(), end);
    } else if (has_normal && !has_color) {
        output->normals_.resize(points_.size());
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(output->points_.begin(), output->normals_.begin()));
        auto end = thrust::copy_if(thrust::make_zip_iterator(thrust::make_tuple(points_.begin(), normals_.begin())),
                                   thrust::make_zip_iterator(thrust::make_tuple(points_.end(), normals_.end())), points_.begin(),
                                   begin, func);
        n_out = thrust::distance(begin, end);
    } else if (!has_normal && has_color) {
        output->colors_.resize(points_.size());
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(output->points_.begin(), output->colors_.begin()));
        auto end = thrust::copy_if(thrust::make_zip_iterator(thrust::make_tuple(points_.begin(), colors_.begin())),
                                   thrust::make_zip_iterator(thrust::make_tuple(points_.end(), colors_.end())), points_.begin(),
                                   begin, func);
        n_out = thrust::distance(begin, end);
    } else if (has_normal && !has_color) {
        output->normals_.resize(points_.size());
        output->colors_.resize(points_.size());
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(output->points_.begin(), output->normals_.begin(), output->colors_.begin()));
        auto end = thrust::copy_if(thrust::make_zip_iterator(thrust::make_tuple(points_.begin(), normals_.begin(), colors_.begin())),
                                   thrust::make_zip_iterator(thrust::make_tuple(points_.end(), normals_.end(), colors_.end())), points_.begin(),
                                   begin, func);
        n_out = thrust::distance(begin, end);
    }
    output->points_.resize(n_out);
    if (has_normal) output->normals_.resize(n_out);
    if (has_color) output->colors_.resize(n_out);
    return output;
}

PointCloud &PointCloud::RemoveNoneFinitePoints(bool remove_nan, bool remove_infinite) {
    bool has_normal = HasNormals();
    bool has_color = HasColors();
    size_t old_point_num = points_.size();
    size_t k = 0;
    check_nan_functor func(remove_nan, remove_infinite);
    if (!has_normal && !has_color) {
        auto end = thrust::remove_if(points_.begin(), points_.end(), func);
        k = thrust::distance(points_.begin(), end);
    } else if (has_normal && !has_color) {
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(points_.begin(), normals_.begin()));
        auto end = thrust::remove_if(begin, thrust::make_zip_iterator(thrust::make_tuple(points_.end(), normals_.end())),
                                     points_.begin(), func);
        k = thrust::distance(begin, end);
    } else if (has_normal && !has_color) {
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(points_.begin(), colors_.begin()));
        auto end = thrust::remove_if(begin, thrust::make_zip_iterator(thrust::make_tuple(points_.end(), colors_.end())),
                                     points_.begin(), func);
        k = thrust::distance(begin, end);
    } else {
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(points_.begin(), normals_.begin(), colors_.begin()));
        auto end = thrust::remove_if(begin, thrust::make_zip_iterator(thrust::make_tuple(points_.end(), normals_.end(), colors_.end())),
                                     points_.begin(), func);
        k = thrust::distance(begin, end);
    }
    points_.resize(k);
    if (has_normal) normals_.resize(k);
    if (has_color) colors_.resize(k);
    utility::LogDebug(
            "[RemoveNoneFinitePoints] {:d} nan points have been removed.",
            (int)(old_point_num - k));
    return *this; 
}
