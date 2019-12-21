#pragma once
#include "cupoch/geometry/geometry.h"
#include "cupoch/utility/eigen.h"
#include <thrust/device_vector.h>

namespace cupoch {
namespace geometry {

class AxisAlignedBoundingBox;
class OrientedBoundingBox;

class Geometry3D : public Geometry {
public:
    ~Geometry3D();

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type type of object based on GeometryType.
    Geometry3D(GeometryType type);

public:
    Geometry3D& Clear() override = 0;
    bool IsEmpty() const override = 0;
    /// Returns min bounds for geometry coordinates.
    virtual Eigen::Vector3f GetMinBound() const = 0;
    /// Returns max bounds for geometry coordinates.
    virtual Eigen::Vector3f GetMaxBound() const = 0;
    /// Returns the center of the geometry coordinates.
    virtual Eigen::Vector3f GetCenter() const = 0;
    /// Returns an axis-aligned bounding box of the geometry.
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const = 0;
    /// \brief Apply transformation (4x4 matrix) to the geometry coordinates.
    virtual Geometry3D& Transform(const Eigen::Matrix4f& transformation) = 0;
    /// \brief Apply translation to the geometry coordinates.
    ///
    /// \param translation A 3D vector to transform the geometry.
    /// \param relative If `true`, the \p translation is directly applied to the
    /// geometry. Otherwise, the geometry center is moved to the \p translation.
    virtual Geometry3D& Translate(const Eigen::Vector3f& translation,
                                  bool relative = true) = 0;
    /// \brief Apply scaling to the geometry coordinates.
    ///
    /// \param scale The scale parameter that is multiplied to the
    /// points/vertices of the geometry.
    /// \param center If `true`, the scale is applied relative to the center of
    /// the geometry. Otherwise, the scale is directly applied to the geometry,
    /// i.e. relative to the origin.
    virtual Geometry3D& Scale(const float scale, bool center = true) = 0;
    /// \brief Apply rotation to the geometry coordinates and normals.
    ///
    /// \param R A 3D vector that either defines the three angles for Euler
    /// rotation, or in the axis-angle representation the normalized vector
    /// defines the axis of rotation and the norm the angle around this axis.
    /// \param center If `true`, the rotation is applied relative to the center
    /// of the geometry. Otherwise, the rotation is directly applied to the
    /// geometry, i.e. relative to the origin.
    virtual Geometry3D& Rotate(const Eigen::Matrix3f& R,
                               bool center = true) = 0;

    Eigen::Vector3f ComputeMinBound(const thrust::device_vector<Eigen::Vector3f>& points) const;
    Eigen::Vector3f ComputeMinBound(cudaStream_t stream, const thrust::device_vector<Eigen::Vector3f>& points) const;

    Eigen::Vector3f ComputeMaxBound(const thrust::device_vector<Eigen::Vector3f>& points) const;
    Eigen::Vector3f ComputeMaxBound(cudaStream_t stream, const thrust::device_vector<Eigen::Vector3f>& points) const;

    Eigen::Vector3f ComputeCenter(const thrust::device_vector<Eigen::Vector3f>& points) const;

    void ResizeAndPaintUniformColor(thrust::device_vector<Eigen::Vector3f>& colors,
                                    const size_t size,
                                    const Eigen::Vector3f& color);

    /// \brief Transforms all points with the transformation matrix.
    ///
    /// \param transformation 4x4 matrix for transformation.
    /// \param points A list of points to be transformed.
    void TransformPoints(const Eigen::Matrix4f& transformation,
                         thrust::device_vector<Eigen::Vector3f>& points);
    void TransformPoints(cudaStream_t stream, const Eigen::Matrix4f& transformation,
                         thrust::device_vector<Eigen::Vector3f>& points);
    /// \brief Transforms the normals with the transformation matrix.
    ///
    /// \param transformation 4x4 matrix for transformation.
    /// \param normals A list of normals to be transformed.
    void TransformNormals(const Eigen::Matrix4f& transformation,
                          thrust::device_vector<Eigen::Vector3f>& normals);
    void TransformNormals(cudaStream_t stream, const Eigen::Matrix4f& transformation,
                          thrust::device_vector<Eigen::Vector3f>& normals);
    /// \brief Apply translation to the geometry coordinates.
    ///
    /// \param translation A 3D vector to transform the geometry.
    /// \param points A list of points to be transformed.
    /// \param relative If `true`, the \p translation is directly applied to the
    /// \points. Otherwise, the center of the \points is moved to the \p
    /// translation.
    void TranslatePoints(const Eigen::Vector3f& translation,
                         thrust::device_vector<Eigen::Vector3f>& points,
                         bool relative) const;
    /// \brief Scale the coordinates of all points by the scaling factor \p
    /// scale.
    ///
    /// \param scale If `true`, the scale is applied relative to the center of
    /// the geometry. Otherwise, the scale is directly applied to the geometry,
    /// i.e. relative to the origin. \param points A list of points to be
    /// transformed. \param center If `true`, then the scale is applied to the
    /// centered geometry.
    void ScalePoints(const float scale,
                     thrust::device_vector<Eigen::Vector3f>& points,
                     bool center) const;
    /// \brief Rotate all points with the rotation matrix \p R.
    ///
    /// \param R A 3D vector that either defines the three angles for Euler
    /// rotation, or in the axis-angle representation the normalized vector
    /// defines the axis of rotation and the norm the angle around this axis.
    /// \param points A list of points to be transformed.
    /// \param center If `true`, the rotation is applied relative to the center
    /// of the geometry. Otherwise, the rotation is directly applied to the
    /// geometry, i.e. relative to the origin.
    void RotatePoints(const Eigen::Matrix3f& R,
                      thrust::device_vector<Eigen::Vector3f>& points,
                      bool center) const;
    void RotatePoints(cudaStream_t stream, const Eigen::Matrix3f& R,
                      thrust::device_vector<Eigen::Vector3f>& points,
                      bool center) const;
    /// \brief Rotate all normals with the rotation matrix \p R.
    ///
    /// \param R A 3D vector that either defines the three angles for Euler
    /// rotation, or in the axis-angle representation the normalized vector
    /// defines the axis of rotation and the norm the angle around this axis.
    /// \param normals A list of normals to be transformed.
    void RotateNormals(const Eigen::Matrix3f& R,
                       thrust::device_vector<Eigen::Vector3f>& normals) const;
    void RotateNormals(cudaStream_t stream, const Eigen::Matrix3f& R,
                       thrust::device_vector<Eigen::Vector3f>& normals) const;
};

}
}