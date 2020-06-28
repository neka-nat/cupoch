#pragma once

#include <thrust/host_vector.h>

#include <Eigen/Core>
#include <memory>

#include "cupoch/geometry/geometry3d.h"
#include "cupoch/geometry/geometry3d_utils.h"
#include "cupoch/utility/device_vector.h"

namespace cupoch {
namespace geometry {

class PointCloud;
class OrientedBoundingBox;
class AxisAlignedBoundingBox;
class TriangleMesh;

class LineSet : public Geometry3D {
public:
    LineSet();
    LineSet(Geometry::GeometryType type);
    LineSet(Geometry::GeometryType type,
            const utility::device_vector<Eigen::Vector3f> &points,
            const utility::device_vector<Eigen::Vector2i> &lines);
    LineSet(const utility::device_vector<Eigen::Vector3f> &points,
            const utility::device_vector<Eigen::Vector2i> &lines);
    LineSet(const thrust::host_vector<Eigen::Vector3f> &points,
            const thrust::host_vector<Eigen::Vector2i> &lines);
    LineSet(const LineSet &other);
    ~LineSet();

    void SetPoints(const thrust::host_vector<Eigen::Vector3f> &points);
    thrust::host_vector<Eigen::Vector3f> GetPoints() const;

    void SetLines(const thrust::host_vector<Eigen::Vector2i> &lines);
    thrust::host_vector<Eigen::Vector2i> GetLines() const;

    void SetColors(const thrust::host_vector<Eigen::Vector3f> &colors);
    thrust::host_vector<Eigen::Vector3f> GetColors() const;

public:
    LineSet &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3f GetMinBound() const override;
    Eigen::Vector3f GetMaxBound() const override;
    Eigen::Vector3f GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    LineSet &Transform(const Eigen::Matrix4f &transformation) override;
    LineSet &Translate(const Eigen::Vector3f &translation,
                       bool relative = true) override;
    LineSet &Scale(const float scale, bool center = true) override;
    LineSet &Rotate(const Eigen::Matrix3f &R, bool center = true) override;

    bool HasPoints() const { return points_.size() > 0; }

    bool HasLines() const { return HasPoints() && lines_.size() > 0; }

    bool HasColors() const {
        return HasLines() && colors_.size() == lines_.size();
    }

    thrust::pair<Eigen::Vector3f, Eigen::Vector3f> GetLineCoordinate(
            size_t line_index) const;

    /// Assigns each line in the LineSet the same color \param color.
    LineSet &PaintUniformColor(const Eigen::Vector3f &color) {
        ResizeAndPaintUniformColor(colors_, lines_.size(), color);
        return *this;
    }

    /// Factory function to create a LineSet from two PointClouds
    /// (\param cloud0, \param cloud1) and a correspondence set
    /// \param correspondences.
    static std::shared_ptr<LineSet> CreateFromPointCloudCorrespondences(
            const PointCloud &cloud0,
            const PointCloud &cloud1,
            const utility::device_vector<thrust::pair<int, int>>
                    &correspondences);

    static std::shared_ptr<LineSet> CreateFromOrientedBoundingBox(
            const OrientedBoundingBox &box);
    static std::shared_ptr<LineSet> CreateFromAxisAlignedBoundingBox(
            const AxisAlignedBoundingBox &box);

    /// Factory function to create a LineSet from edges of a triangle mesh
    /// \param mesh.
    static std::shared_ptr<LineSet> CreateFromTriangleMesh(
            const TriangleMesh &mesh);

public:
    utility::device_vector<Eigen::Vector3f> points_;
    utility::device_vector<Eigen::Vector2i> lines_;
    utility::device_vector<Eigen::Vector3f> colors_;
};

}  // namespace geometry
}  // namespace cupoch