#pragma once
#include <Eigen/Core>
#include "cupoch/geometry/geometry.h"

namespace cupoch {
namespace geometry {

class AxisAlignedBoundingBox;

template <int Dim>
class GeometryBase : public Geometry {
public:
    __host__ __device__ ~GeometryBase(){};  // non-virtual

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type type of object based on GeometryType.
    __host__ __device__ GeometryBase(GeometryType type) : Geometry(type, Dim) {};

public:
    GeometryBase<Dim> &Clear() override = 0;
    bool IsEmpty() const override = 0;
    /// Returns min bounds for geometry coordinates.
    virtual Eigen::Matrix<float, Dim, 1> GetMinBound() const = 0;
    /// Returns max bounds for geometry coordinates.
    virtual Eigen::Matrix<float, Dim, 1> GetMaxBound() const = 0;
    /// Returns the center of the geometry coordinates.
    virtual Eigen::Matrix<float, Dim, 1> GetCenter() const = 0;
    /// Returns an axis-aligned bounding box of the geometry.
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const = 0;
    /// \brief Apply transformation (4x4 matrix) to the geometry coordinates.
    virtual GeometryBase<Dim> &Transform(const Eigen::Matrix<float, Dim + 1, Dim + 1> &transformation) = 0;
    /// \brief Apply translation to the geometry coordinates.
    ///
    /// \param translation A 3D/2D vector to transform the geometry.
    /// \param relative If `true`, the \p translation is directly applied to the
    /// geometry. Otherwise, the geometry center is moved to the \p translation.
    virtual GeometryBase<Dim> &Translate(const Eigen::Matrix<float, Dim, 1> &translation,
                                         bool relative = true) = 0;
    /// \brief Apply scaling to the geometry coordinates.
    ///
    /// \param scale The scale parameter that is multiplied to the
    /// points/vertices of the geometry.
    /// \param center If `true`, the scale is applied relative to the center of
    /// the geometry. Otherwise, the scale is directly applied to the geometry,
    /// i.e. relative to the origin.
    virtual GeometryBase<Dim> &Scale(const float scale, bool center = true) = 0;
    /// \brief Apply rotation to the geometry coordinates and normals.
    ///
    /// \param R A 3D/2D vector that either defines the three angles for Euler
    /// rotation, or in the axis-angle representation the normalized vector
    /// defines the axis of rotation and the norm the angle around this axis.
    /// \param center If `true`, the rotation is applied relative to the center
    /// of the geometry. Otherwise, the rotation is directly applied to the
    /// geometry, i.e. relative to the origin.
    virtual GeometryBase<Dim> &Rotate(const Eigen::Matrix<float, Dim, Dim> &R,
                                      bool center = true) = 0;
};

}  // namespace geometry
}  // namespace cupoch