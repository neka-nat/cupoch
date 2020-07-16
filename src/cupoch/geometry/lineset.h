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
#pragma once

#include <thrust/host_vector.h>

#include <Eigen/Core>
#include <memory>

#include "cupoch/geometry/geometry_base.h"
#include "cupoch/geometry/geometry_utils.h"
#include "cupoch/utility/device_vector.h"

namespace cupoch {
namespace geometry {

class PointCloud;
class OrientedBoundingBox;
class AxisAlignedBoundingBox;
class TriangleMesh;

template <int Dim>
class LineSet : public GeometryBase<Dim> {
public:
    LineSet();
    LineSet(Geometry::GeometryType type);
    LineSet(Geometry::GeometryType type,
            const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
            const utility::device_vector<Eigen::Vector2i> &lines);
    LineSet(const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
            const utility::device_vector<Eigen::Vector2i> &lines);
    LineSet(const thrust::host_vector<Eigen::Matrix<float, Dim, 1>> &points,
            const thrust::host_vector<Eigen::Vector2i> &lines);
    LineSet(const LineSet &other);
    ~LineSet();

    void SetPoints(
            const thrust::host_vector<Eigen::Matrix<float, Dim, 1>> &points);
    thrust::host_vector<Eigen::Matrix<float, Dim, 1>> GetPoints() const;

    void SetLines(const thrust::host_vector<Eigen::Vector2i> &lines);
    thrust::host_vector<Eigen::Vector2i> GetLines() const;

    void SetColors(const thrust::host_vector<Eigen::Vector3f> &colors);
    thrust::host_vector<Eigen::Vector3f> GetColors() const;

public:
    LineSet &Clear() override;
    bool IsEmpty() const override;
    Eigen::Matrix<float, Dim, 1> GetMinBound() const override;
    Eigen::Matrix<float, Dim, 1> GetMaxBound() const override;
    Eigen::Matrix<float, Dim, 1> GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    LineSet<Dim> &Transform(const Eigen::Matrix<float, Dim + 1, Dim + 1>
                                    &transformation) override;
    LineSet<Dim> &Translate(const Eigen::Matrix<float, Dim, 1> &translation,
                            bool relative = true) override;
    LineSet<Dim> &Scale(const float scale, bool center = true) override;
    LineSet<Dim> &Rotate(const Eigen::Matrix<float, Dim, Dim> &R,
                         bool center = true) override;

    bool HasPoints() const { return points_.size() > 0; }

    bool HasLines() const { return HasPoints() && lines_.size() > 0; }

    bool HasColors() const {
        return HasLines() && colors_.size() == lines_.size();
    }

    thrust::pair<Eigen::Matrix<float, Dim, 1>, Eigen::Matrix<float, Dim, 1>>
    GetLineCoordinate(size_t line_index) const;

    /// Assigns each line in the LineSet the same color \param color.
    LineSet &PaintUniformColor(const Eigen::Vector3f &color) {
        ResizeAndPaintUniformColor(colors_, lines_.size(), color);
        return *this;
    }

    /// Factory function to create a LineSet from two PointClouds
    /// (\param cloud0, \param cloud1) and a correspondence set
    /// \param correspondences.
    static std::shared_ptr<LineSet<Dim>> CreateFromPointCloudCorrespondences(
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
    static std::shared_ptr<LineSet<Dim>> CreateFromTriangleMesh(
            const TriangleMesh &mesh);

public:
    utility::device_vector<Eigen::Matrix<float, Dim, 1>> points_;
    utility::device_vector<Eigen::Vector2i> lines_;
    utility::device_vector<Eigen::Vector3f> colors_;
};

}  // namespace geometry
}  // namespace cupoch