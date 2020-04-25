#include <thrust/gather.h>

#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/image.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/helper.h"
#include "cupoch/utility/platform.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

template <class... Args>
struct check_nan_functor {
    check_nan_functor(bool remove_nan, bool remove_infinite)
        : remove_nan_(remove_nan), remove_infinite_(remove_infinite){};
    const bool remove_nan_;
    const bool remove_infinite_;
    __device__ bool operator()(
            const thrust::tuple<Eigen::Vector3f, Args...> &x) const {
        const Eigen::Vector3f &point = thrust::get<0>(x);
        bool is_nan = remove_nan_ &&
                      (isnan(point(0)) || isnan(point(1)) || isnan(point(2)));
        bool is_infinite =
                remove_infinite_ &&
                (isinf(point(0)) || isinf(point(1)) || isinf(point(2)));
        return is_nan || is_infinite;
    }
};

}  // namespace

PointCloud::PointCloud() : Geometry3D(Geometry::GeometryType::PointCloud) {}
PointCloud::PointCloud(const thrust::host_vector<Eigen::Vector3f> &points)
    : Geometry3D(Geometry::GeometryType::PointCloud), points_(points) {}
PointCloud::PointCloud(const utility::device_vector<Eigen::Vector3f> &points)
    : Geometry3D(Geometry::GeometryType::PointCloud), points_(points) {}
PointCloud::PointCloud(const PointCloud &other)
    : Geometry3D(Geometry::GeometryType::PointCloud),
      points_(other.points_),
      normals_(other.normals_),
      colors_(other.colors_) {}

PointCloud::~PointCloud() {}

PointCloud &PointCloud::operator=(const PointCloud &other) {
    points_ = other.points_;
    normals_ = other.normals_;
    colors_ = other.colors_;
    return *this;
}

void PointCloud::SetPoints(const thrust::host_vector<Eigen::Vector3f> &points) {
    points_ = points;
}

thrust::host_vector<Eigen::Vector3f> PointCloud::GetPoints() const {
    thrust::host_vector<Eigen::Vector3f> points = points_;
    return points;
}

void PointCloud::SetNormals(
        const thrust::host_vector<Eigen::Vector3f> &normals) {
    normals_ = normals;
}

thrust::host_vector<Eigen::Vector3f> PointCloud::GetNormals() const {
    thrust::host_vector<Eigen::Vector3f> normals = normals_;
    return normals;
}

void PointCloud::SetColors(const thrust::host_vector<Eigen::Vector3f> &colors) {
    colors_ = colors;
}

thrust::host_vector<Eigen::Vector3f> PointCloud::GetColors() const {
    thrust::host_vector<Eigen::Vector3f> colors = colors_;
    return colors;
}

PointCloud &PointCloud::Clear() {
    points_.clear();
    normals_.clear();
    colors_.clear();
    return *this;
}

bool PointCloud::IsEmpty() const { return !HasPoints(); }

Eigen::Vector3f PointCloud::GetMinBound() const {
    return ComputeMinBound(points_);
}

Eigen::Vector3f PointCloud::GetMaxBound() const {
    return ComputeMaxBound(points_);
}

Eigen::Vector3f PointCloud::GetCenter() const { return ComputeCenter(points_); }

AxisAlignedBoundingBox PointCloud::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(points_);
}

PointCloud &PointCloud::Translate(const Eigen::Vector3f &translation,
                                  bool relative) {
    TranslatePoints(translation, points_, relative);
    return *this;
}

PointCloud &PointCloud::Scale(const float scale, bool center) {
    ScalePoints(scale, points_, center);
    return *this;
}

PointCloud &PointCloud::Rotate(const Eigen::Matrix3f &R, bool center) {
    RotatePoints(utility::GetStream(0), R, points_, center);
    RotateNormals(utility::GetStream(1), R, normals_);
    cudaSafeCall(cudaDeviceSynchronize());
    return *this;
}

PointCloud &PointCloud::operator+=(const PointCloud &cloud) {
    // We do not use std::vector::insert to combine std::vector because it will
    // crash if the pointcloud is added to itself.
    if (cloud.IsEmpty()) return (*this);
    size_t old_vert_num = points_.size();
    size_t add_vert_num = cloud.points_.size();
    size_t new_vert_num = old_vert_num + add_vert_num;
    if ((!HasPoints() || HasNormals()) && cloud.HasNormals()) {
        normals_.resize(new_vert_num);
        thrust::copy(cloud.normals_.begin(),
                     cloud.normals_.end(),
                     normals_.begin() + old_vert_num);
    } else {
        normals_.clear();
    }
    if ((!HasPoints() || HasColors()) && cloud.HasColors()) {
        colors_.resize(new_vert_num);
        thrust::copy(cloud.colors_.begin(),
                     cloud.colors_.end(),
                     colors_.begin() + old_vert_num);
    } else {
        colors_.clear();
    }
    points_.resize(new_vert_num);
    thrust::copy(cloud.points_.begin(), cloud.points_.end(),
                 points_.begin() + old_vert_num);
    return (*this);
}

PointCloud PointCloud::operator+(const PointCloud &cloud) const {
    return (PointCloud(*this) += cloud);
}

PointCloud &PointCloud::NormalizeNormals() {
    thrust::for_each(normals_.begin(), normals_.end(),
                     [] __device__(Eigen::Vector3f & nl) { nl.normalize(); });
    return *this;
}

PointCloud &PointCloud::PaintUniformColor(const Eigen::Vector3f &color) {
    ResizeAndPaintUniformColor(colors_, points_.size(), color);
    return *this;
}

PointCloud &PointCloud::Transform(const Eigen::Matrix4f &transformation) {
    TransformPoints(utility::GetStream(0), transformation, points_);
    TransformNormals(utility::GetStream(1), transformation, normals_);
    cudaSafeCall(cudaDeviceSynchronize());
    return *this;
}

std::shared_ptr<PointCloud> PointCloud::Crop(
        const AxisAlignedBoundingBox &bbox) const {
    if (bbox.IsEmpty()) {
        utility::LogError(
                "[CropPointCloud] AxisAlignedBoundingBox either has zeros "
                "size, or has wrong bounds.");
    }
    return SelectByIndex(bbox.GetPointIndicesWithinBoundingBox(points_));
}

std::shared_ptr<PointCloud> PointCloud::Crop(
        const OrientedBoundingBox &bbox) const {
    if (bbox.IsEmpty()) {
        utility::LogError(
                "[CropPointCloud] AxisAlignedBoundingBox either has zeros "
                "size, or has wrong bounds.");
    }
    return SelectByIndex(bbox.GetPointIndicesWithinBoundingBox(points_));
}

PointCloud &PointCloud::RemoveNoneFinitePoints(bool remove_nan,
                                               bool remove_infinite) {
    bool has_normal = HasNormals();
    bool has_color = HasColors();
    size_t old_point_num = points_.size();
    size_t k = 0;
    if (!has_normal && !has_color) {
        check_nan_functor<> func(remove_nan, remove_infinite);
        auto end = thrust::remove_if(points_.begin(), points_.end(), func);
        k = thrust::distance(points_.begin(), end);
    } else if (has_normal && !has_color) {
        check_nan_functor<Eigen::Vector3f> func(remove_nan, remove_infinite);
        auto begin = make_tuple_iterator(points_.begin(), normals_.begin());
        auto end = thrust::remove_if(
                begin, make_tuple_iterator(points_.end(), normals_.end()),
                func);
        k = thrust::distance(begin, end);
    } else if (!has_normal && has_color) {
        check_nan_functor<Eigen::Vector3f> func(remove_nan, remove_infinite);
        auto begin = make_tuple_iterator(points_.begin(), colors_.begin());
        auto end = thrust::remove_if(
                begin, make_tuple_iterator(points_.end(), colors_.end()), func);
        k = thrust::distance(begin, end);
    } else {
        check_nan_functor<Eigen::Vector3f, Eigen::Vector3f> func(
                remove_nan, remove_infinite);
        auto begin = make_tuple_iterator(points_.begin(), normals_.begin(),
                                         colors_.begin());
        auto end = thrust::remove_if(
                begin,
                make_tuple_iterator(points_.end(), normals_.end(),
                                    colors_.end()),
                func);
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
