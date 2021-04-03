/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
**/
#include <thrust/iterator/discard_iterator.h>
#include <thrust/inner_product.h>

#include <Eigen/Eigenvalues>
#include <numeric>

#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace geometry {

namespace {

struct check_within_oriented_bounding_box_functor {
    check_within_oriented_bounding_box_functor(
            const std::array<Eigen::Vector3f, 8> &box_points)
        : box_points_(box_points){};
    const std::array<Eigen::Vector3f, 8> box_points_;
    __device__ float test_plane(const Eigen::Vector3f &a,
                                const Eigen::Vector3f &b,
                                const Eigen::Vector3f c,
                                const Eigen::Vector3f &x) const {
        Eigen::Matrix3f design;
        design << (b - a), (c - a), (x - a);
        return design.determinant();
    };
    __device__ bool operator()(
            const thrust::tuple<int, Eigen::Vector3f> &x) const {
        const Eigen::Vector3f &point = thrust::get<1>(x);
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

template <int Dim>
struct check_within_axis_aligned_bounding_box_functor {
    check_within_axis_aligned_bounding_box_functor(
            const Eigen::Matrix<float, Dim, 1> *points,
            const Eigen::Matrix<float, Dim, 1> &min_bound,
            const Eigen::Matrix<float, Dim, 1> &max_bound)
        : points_(points), min_bound_(min_bound), max_bound_(max_bound){};
    const Eigen::Matrix<float, Dim, 1> *points_;
    const Eigen::Matrix<float, Dim, 1> min_bound_;
    const Eigen::Matrix<float, Dim, 1> max_bound_;
    __device__ bool operator()(size_t idx) const {
        const Eigen::Matrix<float, Dim, 1> &point = points_[idx];
        #pragma unroll
        for (int i = 0; i < Dim; ++i) {
            if (point(i) < min_bound_(i) || point(i) > max_bound_(i)) {
                return false;
            }
        }
        return true;
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

AxisAlignedBoundingBox<3> OrientedBoundingBox::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox<3>(GetMinBound(), GetMaxBound());
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
    auto begin = make_tuple_iterator(indices.begin(),
                                     thrust::make_discard_iterator());
    auto end = thrust::copy_if(enumerate_begin(points), enumerate_end(points),
            begin, func);
    indices.resize(thrust::distance(begin, end));
    return indices;
}

OrientedBoundingBox OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
        const AxisAlignedBoundingBox<3> &aabox) {
    OrientedBoundingBox obox;
    obox.center_ = aabox.GetCenter();
    obox.extent_ = aabox.GetExtent();
    obox.R_ = Eigen::Matrix3f::Identity();
    return obox;
}


OrientedBoundingBox OrientedBoundingBox::CreateFromPoints(
    const utility::device_vector<Eigen::Vector3f>& points) {
    Eigen::Vector3f mean = thrust::reduce(utility::exec_policy(0)->on(0),
                                          points.begin(), points.end(),
                                          Eigen::Vector3f(0.0, 0.0, 0.0),
                                          thrust::plus<Eigen::Vector3f>());
    mean /= points.size();
    const Eigen::Matrix3f init = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f cov = thrust::transform_reduce(
            utility::exec_policy(0)->on(0),
            points.begin(), points.end(),
            [mean] __device__(const Eigen::Vector3f &pt) -> Eigen::Matrix3f {
                Eigen::Vector3f centered = pt - mean;
                return centered * centered.transpose();
            },
            init, thrust::plus<Eigen::Matrix3f>());
    cov /= points.size();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
    Eigen::Vector3f evals = es.eigenvalues();
    Eigen::Matrix3f R = es.eigenvectors();
    R.col(0) /= R.col(0).norm();
    R.col(1) /= R.col(1).norm();
    R.col(2) /= R.col(2).norm();

    if (evals(1) > evals(0)) {
        std::swap(evals(1), evals(0));
        R.col(1).swap(R.col(0));
    }
    if (evals(2) > evals(0)) {
        std::swap(evals(2), evals(0));
        R.col(2).swap(R.col(0));
    }
    if (evals(2) > evals(1)) {
        std::swap(evals(2), evals(1));
        R.col(2).swap(R.col(1));
    }

    utility::device_vector<Eigen::Vector3f> trans_points = points;
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.block<3, 3>(0, 0) = R.transpose();
    trans.block<3, 1>(0, 2) = -R.transpose() * mean;
    TransformPoints<3>(0, trans, trans_points);
    const auto aabox = AxisAlignedBoundingBox<3>::CreateFromPoints(trans_points);

    OrientedBoundingBox obox;
    obox.center_ = R * aabox.GetCenter() + mean;
    obox.R_ = R;
    obox.extent_ = aabox.GetExtent();

    return obox;
}

template <int Dim>
AxisAlignedBoundingBox<Dim> &AxisAlignedBoundingBox<Dim>::Clear() {
    min_bound_.setZero();
    max_bound_.setZero();
    return *this;
}

template <int Dim>
bool AxisAlignedBoundingBox<Dim>::IsEmpty() const { return Volume() <= 0; }

template <int Dim>
Eigen::Matrix<float, Dim, 1> AxisAlignedBoundingBox<Dim>::GetMinBound() const {
    return min_bound_;
}

template <int Dim>
Eigen::Matrix<float, Dim, 1> AxisAlignedBoundingBox<Dim>::GetMaxBound() const {
    return max_bound_;
}

template <int Dim>
Eigen::Matrix<float, Dim, 1> AxisAlignedBoundingBox<Dim>::GetCenter() const {
    return (min_bound_ + max_bound_) * 0.5;
}

template <int Dim>
AxisAlignedBoundingBox<Dim> AxisAlignedBoundingBox<Dim>::GetAxisAlignedBoundingBox()
        const {
    return *this;
}

template <>
template <>
OrientedBoundingBox AxisAlignedBoundingBox<3>::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(*this);
}

template <int Dim>
AxisAlignedBoundingBox<Dim> &AxisAlignedBoundingBox<Dim>::Transform(
        const Eigen::Matrix<float, Dim + 1, Dim + 1> &transformation) {
    utility::LogError(
            "A general transform of a AxisAlignedBoundingBox would not be axis "
            "aligned anymore, convert it to a OrientedBoundingBox first");
    return *this;
}

template <int Dim>
AxisAlignedBoundingBox<Dim> &AxisAlignedBoundingBox<Dim>::Translate(
        const Eigen::Matrix<float, Dim, 1> &translation, bool relative) {
    if (relative) {
        min_bound_ += translation;
        max_bound_ += translation;
    } else {
        const Eigen::Matrix<float, Dim, 1> half_extent = GetHalfExtent();
        min_bound_ = translation - half_extent;
        max_bound_ = translation + half_extent;
    }
    return *this;
}

template <int Dim>
AxisAlignedBoundingBox<Dim> &AxisAlignedBoundingBox<Dim>::Scale(const float scale,
                                                                bool center) {
    if (center) {
        Eigen::Matrix<float, Dim, 1> center = GetCenter();
        min_bound_ = center + scale * (min_bound_ - center);
        max_bound_ = center + scale * (max_bound_ - center);
    } else {
        min_bound_ *= scale;
        max_bound_ *= scale;
    }
    return *this;
}

template <int Dim>
AxisAlignedBoundingBox<Dim> &AxisAlignedBoundingBox<Dim>::Rotate(
        const Eigen::Matrix<float, Dim, Dim> &rotation, bool center) {
    utility::LogError(
            "A rotation of a AxisAlignedBoundingBox would not be axis aligned "
            "anymore, convert it to an OrientedBoundingBox first");
    return *this;
}

template <>
template <>
std::string AxisAlignedBoundingBox<3>::GetPrintInfo() const {
    return fmt::format("[({:.4f}, {:.4f}, {:.4f}) - ({:.4f}, {:.4f}, {:.4f})]",
                       min_bound_(0), min_bound_(1), min_bound_(2),
                       max_bound_(0), max_bound_(1), max_bound_(2));
}

template <int Dim>
AxisAlignedBoundingBox<Dim> &AxisAlignedBoundingBox<Dim>::operator+=(
        const AxisAlignedBoundingBox<Dim> &other) {
    if (IsEmpty()) {
        min_bound_ = other.min_bound_;
        max_bound_ = other.max_bound_;
    } else if (!other.IsEmpty()) {
        min_bound_ = min_bound_.array().min(other.min_bound_.array()).matrix();
        max_bound_ = max_bound_.array().max(other.max_bound_.array()).matrix();
    }
    return *this;
}

template <int Dim>
AxisAlignedBoundingBox<Dim> AxisAlignedBoundingBox<Dim>::CreateFromPoints(
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points) {
    AxisAlignedBoundingBox box;
    if (points.empty()) {
        box.min_bound_ = Eigen::Matrix<float, Dim, 1>::Zero();
        box.max_bound_ = Eigen::Matrix<float, Dim, 1>::Zero();
    } else {
        box.min_bound_ = ComputeBound<Dim, thrust::elementwise_minimum<Eigen::Matrix<float, Dim, 1>>>(utility::GetStream(0), points);
        box.max_bound_ = ComputeBound<Dim, thrust::elementwise_maximum<Eigen::Matrix<float, Dim, 1>>>(utility::GetStream(1), points);
    }
    return box;
}

template <int Dim>
float AxisAlignedBoundingBox<Dim>::Volume() const { return GetExtent().prod(); }

template <>
template <>
std::array<Eigen::Vector3f, 8> AxisAlignedBoundingBox<3>::GetBoxPoints() const {
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

template <int Dim>
utility::device_vector<size_t>
AxisAlignedBoundingBox<Dim>::GetPointIndicesWithinBoundingBox(
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points) const {
    utility::device_vector<size_t> indices(points.size());
    check_within_axis_aligned_bounding_box_functor<Dim> func(
            thrust::raw_pointer_cast(points.data()), min_bound_, max_bound_);
    auto end = thrust::copy_if(thrust::make_counting_iterator<size_t>(0),
                               thrust::make_counting_iterator(points.size()),
                               indices.begin(), func);
    indices.resize(thrust::distance(indices.begin(), end));
    return indices;
}

template class AxisAlignedBoundingBox<2>;
template class AxisAlignedBoundingBox<3>;

}
}