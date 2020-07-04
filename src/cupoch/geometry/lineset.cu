#include <numeric>

#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/lineset.h"

using namespace cupoch;
using namespace cupoch::geometry;

template <int Dim>
LineSet<Dim>::LineSet() : GeometryBase<Dim>(Geometry::GeometryType::LineSet) {}

template <int Dim>
LineSet<Dim>::LineSet(Geometry::GeometryType type) : GeometryBase<Dim>(type) {}

template <int Dim>
LineSet<Dim>::LineSet(
        Geometry::GeometryType type,
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
        const utility::device_vector<Eigen::Vector2i> &lines)
    : GeometryBase<Dim>(type), points_(points), lines_(lines) {}

template <int Dim>
LineSet<Dim>::LineSet(
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
        const utility::device_vector<Eigen::Vector2i> &lines)
    : GeometryBase<Dim>(Geometry::GeometryType::LineSet),
      points_(points),
      lines_(lines) {}

template <int Dim>
LineSet<Dim>::LineSet(
        const thrust::host_vector<Eigen::Matrix<float, Dim, 1>> &points,
        const thrust::host_vector<Eigen::Vector2i> &lines)
    : GeometryBase<Dim>(Geometry::GeometryType::LineSet),
      points_(points),
      lines_(lines) {}

template <int Dim>
LineSet<Dim>::LineSet(const LineSet &other)
    : GeometryBase<Dim>(Geometry::GeometryType::LineSet),
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
    return ComputeMinBound<Dim>(points_);
}

template <int Dim>
Eigen::Matrix<float, Dim, 1> LineSet<Dim>::GetMaxBound() const {
    return ComputeMaxBound<Dim>(points_);
}

template <int Dim>
Eigen::Matrix<float, Dim, 1> LineSet<Dim>::GetCenter() const {
    return ComputeCenter<Dim>(points_);
}

template <int Dim>
AxisAlignedBoundingBox LineSet<Dim>::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(points_);
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

template class LineSet<2>;
template class LineSet<3>;