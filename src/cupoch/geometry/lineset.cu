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
#include <numeric>

#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/lineset.h"

#include "cupoch/utility/helper.h"
#include "cupoch/utility/eigen.h"

using namespace cupoch;
using namespace cupoch::geometry;

template <int Dim>
LineSet<Dim>::LineSet()
    : GeometryBaseXD<Dim>(Geometry::GeometryType::LineSet) {}

template <int Dim>
LineSet<Dim>::LineSet(Geometry::GeometryType type)
    : GeometryBaseXD<Dim>(type) {}

template <int Dim>
LineSet<Dim>::LineSet(
        Geometry::GeometryType type,
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
        const utility::device_vector<Eigen::Vector2i> &lines)
    : GeometryBaseXD<Dim>(type), points_(points), lines_(lines) {}

template <int Dim>
LineSet<Dim>::LineSet(
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
        const utility::device_vector<Eigen::Vector2i> &lines)
    : GeometryBaseXD<Dim>(Geometry::GeometryType::LineSet),
      points_(points),
      lines_(lines) {}

template <int Dim>
LineSet<Dim>::LineSet(
        const utility::pinned_host_vector<Eigen::Matrix<float, Dim, 1>> &points,
        const utility::pinned_host_vector<Eigen::Vector2i> &lines)
    : GeometryBaseXD<Dim>(Geometry::GeometryType::LineSet),
      points_(points),
      lines_(lines) {}

template <int Dim>
LineSet<Dim>::LineSet(
        const thrust::host_vector<Eigen::Matrix<float, Dim, 1>> &points,
        const thrust::host_vector<Eigen::Vector2i> &lines)
    : GeometryBaseXD<Dim>(Geometry::GeometryType::LineSet),
      points_(points),
      lines_(lines) {}

template <int Dim>
LineSet<Dim>::LineSet(const std::vector<Eigen::Matrix<float, Dim, 1>> &points,
                     const std::vector<Eigen::Vector2i> &lines)
    : GeometryBaseXD<Dim>(Geometry::GeometryType::LineSet),
      points_(points),
      lines_(lines) {}

template <int Dim>
LineSet<Dim>::LineSet(const std::vector<Eigen::Matrix<float, Dim, 1>> &path)
    : GeometryBaseXD<Dim>(Geometry::GeometryType::LineSet) {
    utility::pinned_host_vector<Eigen::Matrix<float, Dim, 1>> points;
    utility::pinned_host_vector<Eigen::Vector2i> lines;
    for (int i = 0; i < path.size() - 1; ++i) {
        points.push_back(path[i]);
        lines.push_back(Eigen::Vector2i(i, i + 1));
    }
    points.push_back(path.back());
    points_.resize(points.size());
    lines_.resize(lines.size());
    copy_host_to_device(points, points_);
    copy_host_to_device(lines, lines_);
    cudaSafeCall(cudaDeviceSynchronize());
}

template <int Dim>
LineSet<Dim>::LineSet(const LineSet &other)
    : GeometryBaseXD<Dim>(Geometry::GeometryType::LineSet),
      points_(other.points_),
      lines_(other.lines_),
      colors_(other.colors_) {}

template <int Dim>
LineSet<Dim>::~LineSet() {}

template <int Dim>
void LineSet<Dim>::SetPoints(
        const thrust::host_vector<Eigen::Matrix<float, Dim, 1>> &points) {
    points_ = points;
}

template <int Dim>
void LineSet<Dim>::SetPoints(
        const std::vector<Eigen::Matrix<float, Dim, 1>> &points) {
    points_.resize(points.size());
    copy_host_to_device(points, points_);
}

template <int Dim>
thrust::host_vector<Eigen::Matrix<float, Dim, 1>> LineSet<Dim>::GetPoints()
        const {
    thrust::host_vector<Eigen::Matrix<float, Dim, 1>> points = points_;
    return points;
}

template <int Dim>
void LineSet<Dim>::SetLines(const thrust::host_vector<Eigen::Vector2i> &lines) {
    lines_ = lines;
}

template <int Dim>
void LineSet<Dim>::SetLines(const std::vector<Eigen::Vector2i> &lines) {
    lines_.resize(lines.size());
    copy_host_to_device(lines, lines_);
}

template <int Dim>
thrust::host_vector<Eigen::Vector2i> LineSet<Dim>::GetLines() const {
    thrust::host_vector<Eigen::Vector2i> lines = lines_;
    return lines;
}

template <int Dim>
void LineSet<Dim>::SetColors(
        const thrust::host_vector<Eigen::Vector3f> &colors) {
    colors_ = colors;
}

template <int Dim>
void LineSet<Dim>::SetColors(const std::vector<Eigen::Vector3f> &colors) {
    colors_.resize(colors.size());
    copy_host_to_device(colors, colors_);
}

template <int Dim>
thrust::host_vector<Eigen::Vector3f> LineSet<Dim>::GetColors() const {
    thrust::host_vector<Eigen::Vector3f> colors = colors_;
    return colors;
}

template <int Dim>
LineSet<Dim> &LineSet<Dim>::Clear() {
    points_.clear();
    lines_.clear();
    colors_.clear();
    return *this;
}

template <int Dim>
bool LineSet<Dim>::IsEmpty() const {
    return !HasPoints();
}

template <int Dim>
Eigen::Matrix<float, Dim, 1> LineSet<Dim>::GetMinBound() const {
    return utility::ComputeMinBound<Dim>(points_);
}

template <int Dim>
Eigen::Matrix<float, Dim, 1> LineSet<Dim>::GetMaxBound() const {
    return utility::ComputeMaxBound<Dim>(points_);
}

template <int Dim>
Eigen::Matrix<float, Dim, 1> LineSet<Dim>::GetCenter() const {
    return utility::ComputeCenter<Dim>(points_);
}

template <int Dim>
AxisAlignedBoundingBox<Dim> LineSet<Dim>::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox<Dim>::CreateFromPoints(points_);
}

template <int Dim>
LineSet<Dim> &LineSet<Dim>::Transform(
        const Eigen::Matrix<float, Dim + 1, Dim + 1> &transformation) {
    TransformPoints<Dim>(transformation, points_);
    return *this;
}

template <int Dim>
LineSet<Dim> &LineSet<Dim>::Translate(
        const Eigen::Matrix<float, Dim, 1> &translation, bool relative) {
    TranslatePoints<Dim>(translation, points_, relative);
    return *this;
}

template <int Dim>
LineSet<Dim> &LineSet<Dim>::Scale(const float scale, bool center) {
    ScalePoints<Dim>(scale, points_, center);
    return *this;
}

template <int Dim>
LineSet<Dim> &LineSet<Dim>::Rotate(const Eigen::Matrix<float, Dim, Dim> &R,
                                   bool center) {
    RotatePoints<Dim>(R, points_, center);
    return *this;
}

template <int Dim>
thrust::pair<Eigen::Matrix<float, Dim, 1>, Eigen::Matrix<float, Dim, 1>>
LineSet<Dim>::GetLineCoordinate(size_t line_index) const {
    const Eigen::Vector2i idxs = lines_[line_index];
    return thrust::make_pair(points_[idxs[0]], points_[idxs[1]]);
}

template <int Dim>
LineSet<Dim> &LineSet<Dim>::PaintUniformColor(const Eigen::Vector3f &color) {
    ResizeAndPaintUniformColor(colors_, lines_.size(), color);
    return *this;
}

template <int Dim>
float LineSet<Dim>::GetMaxLineLength() const {
    return thrust::transform_reduce(
            make_tuple_iterator(
                    thrust::make_permutation_iterator(
                            points_.begin(),
                            thrust::make_transform_iterator(
                                    lines_.begin(),
                                    element_get_functor<Eigen::Vector2i, 0>())),
                    thrust::make_permutation_iterator(
                            points_.begin(),
                            thrust::make_transform_iterator(
                                    lines_.begin(),
                                    element_get_functor<Eigen::Vector2i, 1>()))),
            make_tuple_iterator(
                    thrust::make_permutation_iterator(
                            points_.begin(),
                            thrust::make_transform_iterator(
                                    lines_.end(),
                                    element_get_functor<Eigen::Vector2i, 0>())),
                    thrust::make_permutation_iterator(
                            points_.begin(),
                            thrust::make_transform_iterator(
                                    lines_.end(),
                                    element_get_functor<Eigen::Vector2i, 1>()))),
            [] __device__(
                    const thrust::tuple<Eigen::Matrix<float, Dim, 1>,
                                        Eigen::Matrix<float, Dim, 1>> &ppair) -> float {
                return (thrust::get<0>(ppair) - thrust::get<1>(ppair)).norm();
            },
            0.0f, thrust::maximum<float>());
}

template <int Dim>
LineSet<Dim> &LineSet<Dim>::PaintIndexedColor(
        const utility::device_vector<size_t> &indices,
        const Eigen::Vector3f &color) {
    if (colors_.empty()) {
        colors_.resize(lines_.size());
        thrust::fill(colors_.begin(), colors_.end(), DEFAULT_LINE_COLOR);
    }
    thrust::for_each(
            thrust::make_permutation_iterator(colors_.begin(), indices.begin()),
            thrust::make_permutation_iterator(colors_.begin(), indices.end()),
            [tc = color] __device__(Eigen::Vector3f & sc) { sc = tc; });
    return *this;
}

template class LineSet<2>;
template class LineSet<3>;