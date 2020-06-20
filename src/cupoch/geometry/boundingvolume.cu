#include <Eigen/Eigenvalues>
#include <numeric>

#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/platform.h"

#include <thrust/iterator/discard_iterator.h>

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct check_within_oriented_bounding_box_functor {
    check_within_oriented_bounding_box_functor(
            const std::array<Eigen::Vector3f, 8> &box_points)
        : box_points_(box_points) {};
    const std::array<Eigen::Vector3f, 8> box_points_;
    __device__ float test_plane(const Eigen::Vector3f &a,
                                const Eigen::Vector3f &b,
                                const Eigen::Vector3f c,
                                const Eigen::Vector3f &x) const {
        Eigen::Matrix3f design;
        design << (b - a), (c - a), (x - a);
        return design.determinant();
    };
    __device__ bool operator()(const thrust::tuple<int, Eigen::Vector3f>& x) const {
        const Eigen::Vector3f& point = thrust::get<1>(x);
        return (test_plane(box_points_[0], box_points_[1], box_points_[3],
                           point) <= 0 &&
                test_plane(box_points_[0], box_points_[5], box_points_[3],
                           point) >= 0 &&
                test_plane(box_points_[2], box_points_[5], box_points_[7],
                           point) <= 0 &&
                test_plane(box_points_[1], box_points_[4], box_points_[7],
                           point) >= 0 &&
                test_plane(box_points_[3], box_points_[4], box_points_[5],
                           point) <= 0 &&
                test_plane(box_points_[0], box_points_[1], box_points_[7],
                           point) >= 0);
    }
};

struct check_within_axis_aligned_bounding_box_functor {
    check_within_axis_aligned_bounding_box_functor(
            const Eigen::Vector3f *points,
            const Eigen::Vector3f &min_bound,
            const Eigen::Vector3f &max_bound)
        : points_(points), min_bound_(min_bound), max_bound_(max_bound){};
    const Eigen::Vector3f *points_;
    const Eigen::Vector3f min_bound_;
    const Eigen::Vector3f max_bound_;
    __device__ bool operator()(size_t idx) const {
        const Eigen::Vector3f &point = points_[idx];
        return (point(0) >= min_bound_(0) && point(0) <= max_bound_(0) &&
                point(1) >= min_bound_(1) && point(1) <= max_bound_(1) &&
                point(2) >= min_bound_(2) && point(2) <= max_bound_(2));
    }
};

}  // namespace

OrientedBoundingBox &OrientedBoundingBox::Clear() {
    center_.setZero();
    extent_.setZero();
    R_ = Eigen::Matrix3f::Identity();
    color_.setZero();
    return *this;
}

bool OrientedBoundingBox::IsEmpty() const { return Volume() <= 0; }

Eigen::Vector3f OrientedBoundingBox::GetMinBound() const {
    const Eigen::Vector3f size = (R_ * extent_).array().abs().matrix();
    return center_ - 0.5 * size;
}

Eigen::Vector3f OrientedBoundingBox::GetMaxBound() const {
    const Eigen::Vector3f size = (R_ * extent_).array().abs().matrix();
    return center_ + 0.5 * size;
}

Eigen::Vector3f OrientedBoundingBox::GetCenter() const { return center_; }

AxisAlignedBoundingBox OrientedBoundingBox::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox(GetMinBound(), GetMaxBound());
}

OrientedBoundingBox OrientedBoundingBox::GetOrientedBoundingBox() const {
    return *this;
}

OrientedBoundingBox &OrientedBoundingBox::Transform(
        const Eigen::Matrix4f &transformation) {
    utility::LogError(
            "A general transform of an OrientedBoundingBox is not implemented. "
            "Call Translate, Scale, and Rotate.");
    return *this;
}

OrientedBoundingBox &OrientedBoundingBox::Translate(
        const Eigen::Vector3f &translation, bool relative) {
    if (relative) {
        center_ += translation;
    } else {
        center_ = translation;
    }
    return *this;
}

OrientedBoundingBox &OrientedBoundingBox::Scale(const float scale,
                                                bool center) {
    if (center) {
        extent_ *= scale;
    } else {
        center_ *= scale;
        extent_ *= scale;
    }
    return *this;
}

OrientedBoundingBox &OrientedBoundingBox::Rotate(const Eigen::Matrix3f &R,
                                                 bool center) {
    if (center) {
        R_ *= R;
    } else {
        center_ = R * center_;
        R_ *= R;
    }
    return *this;
}

float OrientedBoundingBox::Volume() const {
    return extent_(0) * extent_(1) * extent_(2);
}

std::array<Eigen::Vector3f, 8> OrientedBoundingBox::GetBoxPoints() const {
    Eigen::Vector3f x_axis = R_ * Eigen::Vector3f(extent_(0) / 2, 0, 0);
    Eigen::Vector3f y_axis = R_ * Eigen::Vector3f(0, extent_(1) / 2, 0);
    Eigen::Vector3f z_axis = R_ * Eigen::Vector3f(0, 0, extent_(2) / 2);
    std::array<Eigen::Vector3f, 8> points;
    points[0] = center_ - x_axis - y_axis - z_axis;
    points[1] = center_ + x_axis - y_axis - z_axis;
    points[2] = center_ - x_axis + y_axis - z_axis;
    points[3] = center_ - x_axis - y_axis + z_axis;
    points[4] = center_ + x_axis + y_axis + z_axis;
    points[5] = center_ - x_axis + y_axis + z_axis;
    points[6] = center_ + x_axis - y_axis + z_axis;
    points[7] = center_ + x_axis + y_axis - z_axis;
    return points;
}

utility::device_vector<size_t>
OrientedBoundingBox::GetPointIndicesWithinBoundingBox(
        const utility::device_vector<Eigen::Vector3f> &points) const {
    auto box_points = GetBoxPoints();
    check_within_oriented_bounding_box_functor func(box_points);
    utility::device_vector<size_t> indices(points.size());
    auto begin = make_tuple_iterator(indices.begin(), thrust::make_discard_iterator());
    auto end = thrust::copy_if(make_tuple_iterator(thrust::make_counting_iterator(0), points.begin()),
                               make_tuple_iterator(thrust::make_counting_iterator<int>(points.size()), points.end()),
                               begin, func);
    indices.resize(thrust::distance(begin, end));
    return indices;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
        const AxisAlignedBoundingBox &aabox) {
    OrientedBoundingBox obox;
    obox.center_ = aabox.GetCenter();
    obox.extent_ = aabox.GetExtent();
    obox.R_ = Eigen::Matrix3f::Identity();
    return obox;
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Clear() {
    min_bound_.setZero();
    max_bound_.setZero();
    return *this;
}

bool AxisAlignedBoundingBox::IsEmpty() const { return Volume() <= 0; }
Eigen::Vector3f AxisAlignedBoundingBox::GetMinBound() const {
    return min_bound_;
}

Eigen::Vector3f AxisAlignedBoundingBox::GetMaxBound() const {
    return max_bound_;
}

Eigen::Vector3f AxisAlignedBoundingBox::GetCenter() const {
    return (min_bound_ + max_bound_) * 0.5;
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::GetAxisAlignedBoundingBox()
        const {
    return *this;
}

OrientedBoundingBox AxisAlignedBoundingBox::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(*this);
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Transform(
        const Eigen::Matrix4f &transformation) {
    utility::LogError(
            "A general transform of a AxisAlignedBoundingBox would not be axis "
            "aligned anymore, convert it to a OrientedBoundingBox first");
    return *this;
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Translate(
        const Eigen::Vector3f &translation, bool relative) {
    if (relative) {
        min_bound_ += translation;
        max_bound_ += translation;
    } else {
        const Eigen::Vector3f half_extent = GetHalfExtent();
        min_bound_ = translation - half_extent;
        max_bound_ = translation + half_extent;
    }
    return *this;
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Scale(const float scale,
                                                      bool center) {
    if (center) {
        Eigen::Vector3f center = GetCenter();
        min_bound_ = center + scale * (min_bound_ - center);
        max_bound_ = center + scale * (max_bound_ - center);
    } else {
        min_bound_ *= scale;
        max_bound_ *= scale;
    }
    return *this;
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::Rotate(
        const Eigen::Matrix3f &rotation, bool center) {
    utility::LogError(
            "A rotation of a AxisAlignedBoundingBox would not be axis aligned "
            "anymore, convert it to an OrientedBoundingBox first");
    return *this;
}

std::string AxisAlignedBoundingBox::GetPrintInfo() const {
    return fmt::format("[({:.4f}, {:.4f}, {:.4f}) - ({:.4f}, {:.4f}, {:.4f})]",
                       min_bound_(0), min_bound_(1), min_bound_(2),
                       max_bound_(0), max_bound_(1), max_bound_(2));
}

AxisAlignedBoundingBox &AxisAlignedBoundingBox::operator+=(
        const AxisAlignedBoundingBox &other) {
    if (IsEmpty()) {
        min_bound_ = other.min_bound_;
        max_bound_ = other.max_bound_;
    } else if (!other.IsEmpty()) {
        min_bound_ = min_bound_.array().min(other.min_bound_.array()).matrix();
        max_bound_ = max_bound_.array().max(other.max_bound_.array()).matrix();
    }
    return *this;
}

AxisAlignedBoundingBox AxisAlignedBoundingBox::CreateFromPoints(
        const utility::device_vector<Eigen::Vector3f> &points) {
    AxisAlignedBoundingBox box;
    if (points.empty()) {
        box.min_bound_ = Eigen::Vector3f(0.0, 0.0, 0.0);
        box.max_bound_ = Eigen::Vector3f(0.0, 0.0, 0.0);
    } else {
        box.min_bound_ = box.ComputeMinBound(utility::GetStream(0), points);
        box.max_bound_ = box.ComputeMaxBound(utility::GetStream(1), points);
    }
    return box;
}

float AxisAlignedBoundingBox::Volume() const { return GetExtent().prod(); }

std::array<Eigen::Vector3f, 8> AxisAlignedBoundingBox::GetBoxPoints() const {
    std::array<Eigen::Vector3f, 8> points;
    Eigen::Vector3f extent = GetExtent();
    points[0] = min_bound_;
    points[1] = min_bound_ + Eigen::Vector3f(extent(0), 0, 0);
    points[2] = min_bound_ + Eigen::Vector3f(0, extent(1), 0);
    points[3] = min_bound_ + Eigen::Vector3f(0, 0, extent(2));
    points[4] = max_bound_;
    points[5] = max_bound_ - Eigen::Vector3f(extent(0), 0, 0);
    points[6] = max_bound_ - Eigen::Vector3f(0, extent(1), 0);
    points[7] = max_bound_ - Eigen::Vector3f(0, 0, extent(2));
    return points;
}

utility::device_vector<size_t>
AxisAlignedBoundingBox::GetPointIndicesWithinBoundingBox(
        const utility::device_vector<Eigen::Vector3f> &points) const {
    utility::device_vector<size_t> indices(points.size());
    check_within_axis_aligned_bounding_box_functor func(
            thrust::raw_pointer_cast(points.data()), min_bound_, max_bound_);
    auto end = thrust::copy_if(thrust::make_counting_iterator<size_t>(0),
                               thrust::make_counting_iterator(points.size()),
                               indices.begin(), func);
    indices.resize(thrust::distance(indices.begin(), end));
    return indices;
}