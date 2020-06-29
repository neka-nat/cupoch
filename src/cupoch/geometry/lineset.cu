#include <numeric>

#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/lineset.h"

using namespace cupoch;
using namespace cupoch::geometry;

LineSet::LineSet() : GeometryBase<3>(Geometry::GeometryType::LineSet) {}

LineSet::LineSet(Geometry::GeometryType type) : GeometryBase<3>(type) {}

LineSet::LineSet(Geometry::GeometryType type,
                 const utility::device_vector<Eigen::Vector3f> &points,
                 const utility::device_vector<Eigen::Vector2i> &lines)
    : GeometryBase<3>(type),
      points_(points),
      lines_(lines) {}

LineSet::LineSet(const utility::device_vector<Eigen::Vector3f> &points,
                 const utility::device_vector<Eigen::Vector2i> &lines)
    : GeometryBase<3>(Geometry::GeometryType::LineSet),
      points_(points),
      lines_(lines) {}

LineSet::LineSet(const thrust::host_vector<Eigen::Vector3f> &points,
                 const thrust::host_vector<Eigen::Vector2i> &lines)
    : GeometryBase<3>(Geometry::GeometryType::LineSet),
      points_(points),
      lines_(lines) {}

LineSet::LineSet(const LineSet &other)
    : GeometryBase<3>(Geometry::GeometryType::LineSet),
      points_(other.points_),
      lines_(other.lines_),
      colors_(other.colors_) {}

LineSet::~LineSet() {}

void LineSet::SetPoints(const thrust::host_vector<Eigen::Vector3f> &points) {
    points_ = points;
}

thrust::host_vector<Eigen::Vector3f> LineSet::GetPoints() const {
    thrust::host_vector<Eigen::Vector3f> points = points_;
    return points;
}

void LineSet::SetLines(const thrust::host_vector<Eigen::Vector2i> &lines) {
    lines_ = lines;
}

thrust::host_vector<Eigen::Vector2i> LineSet::GetLines() const {
    thrust::host_vector<Eigen::Vector2i> lines = lines_;
    return lines;
}

void LineSet::SetColors(const thrust::host_vector<Eigen::Vector3f> &colors) {
    colors_ = colors;
}

thrust::host_vector<Eigen::Vector3f> LineSet::GetColors() const {
    thrust::host_vector<Eigen::Vector3f> colors = colors_;
    return colors;
}

LineSet &LineSet::Clear() {
    points_.clear();
    lines_.clear();
    colors_.clear();
    return *this;
}

bool LineSet::IsEmpty() const { return !HasPoints(); }

Eigen::Vector3f LineSet::GetMinBound() const {
    return ComputeMinBound(points_);
}

Eigen::Vector3f LineSet::GetMaxBound() const {
    return ComputeMaxBound(points_);
}

Eigen::Vector3f LineSet::GetCenter() const { return ComputeCenter(points_); }

AxisAlignedBoundingBox LineSet::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(points_);
}

LineSet &LineSet::Transform(const Eigen::Matrix4f &transformation) {
    TransformPoints(transformation, points_);
    return *this;
}

LineSet &LineSet::Translate(const Eigen::Vector3f &translation, bool relative) {
    TranslatePoints(translation, points_, relative);
    return *this;
}

LineSet &LineSet::Scale(const float scale, bool center) {
    ScalePoints(scale, points_, center);
    return *this;
}

LineSet &LineSet::Rotate(const Eigen::Matrix3f &R, bool center) {
    RotatePoints(R, points_, center);
    return *this;
}

thrust::pair<Eigen::Vector3f, Eigen::Vector3f> LineSet::GetLineCoordinate(
        size_t line_index) const {
    const Eigen::Vector2i idxs = lines_[line_index];
    return thrust::make_pair(points_[idxs[0]], points_[idxs[1]]);
}