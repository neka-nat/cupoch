#pragma once
#include <Eigen/Core>
#include "cupoch/geometry/geometry.h"

namespace cupoch {
namespace geometry {

class AxisAlignedBoundingBox;
class OrientedBoundingBox;

class Geometry3D : public Geometry {
public:
    __host__ __device__ ~Geometry3D(){};  // non-virtual

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type type of object based on GeometryType.
    __host__ __device__ Geometry3D(GeometryType type) : Geometry(type, 3) {};

public:
    Geometry3D &Clear() override = 0;
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
    virtual Geometry3D &Transform(const Eigen::Matrix4f &transformation) = 0;
    /// \brief Apply translation to the geometry coordinates.
    ///
    /// \param translation A 3D vector to transform the geometry.
    /// \param relative If `true`, the \p translation is directly applied to the
    /// geometry. Otherwise, the geometry center is moved to the \p translation.
    virtual Geometry3D &Translate(const Eigen::Vector3f &translation,
                                  bool relative = true) = 0;
    /// \brief Apply scaling to the geometry coordinates.
    ///
    /// \param scale The scale parameter that is multiplied to the
    /// points/vertices of the geometry.
    /// \param center If `true`, the scale is applied relative to the center of
    /// the geometry. Otherwise, the scale is directly applied to the geometry,
    /// i.e. relative to the origin.
    virtual Geometry3D &Scale(const float scale, bool center = true) = 0;
    /// \brief Apply rotation to the geometry coordinates and normals.
    ///
    /// \param R A 3D vector that either defines the three angles for Euler
    /// rotation, or in the axis-angle representation the normalized vector
    /// defines the axis of rotation and the norm the angle around this axis.
    /// \param center If `true`, the rotation is applied relative to the center
    /// of the geometry. Otherwise, the rotation is directly applied to the
    /// geometry, i.e. relative to the origin.
    virtual Geometry3D &Rotate(const Eigen::Matrix3f &R,
                               bool center = true) = 0;
};

}  // namespace geometry
}  // namespace cupoch