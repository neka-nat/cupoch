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
#include <Eigen/Core>

#include "cupoch/geometry/geometry.h"

namespace cupoch {
namespace geometry {

class AxisAlignedBoundingBox;

template <typename VectorT, typename MatrixT, typename TransformT>
class GeometryBase : public Geometry {
public:
    __host__ __device__ ~GeometryBase(){};  // non-virtual

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type type of object based on GeometryType.
    __host__ __device__ GeometryBase(GeometryType type, int dimension = VectorT::SizeAtCompileTime)
    : Geometry(type, dimension){};

public:
    GeometryBase<VectorT, MatrixT, TransformT> &Clear() override = 0;
    bool IsEmpty() const override = 0;
    /// Returns min bounds for geometry coordinates.
    virtual VectorT GetMinBound() const = 0;
    /// Returns max bounds for geometry coordinates.
    virtual VectorT GetMaxBound() const = 0;
    /// Returns the center of the geometry coordinates.
    virtual VectorT GetCenter() const = 0;
    /// Returns an axis-aligned bounding box of the geometry.
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const = 0;
    /// \brief Apply transformation (4x4 matrix) to the geometry coordinates.
    virtual GeometryBase<VectorT, MatrixT, TransformT> &Transform(
            const TransformT &transformation) = 0;
    /// \brief Apply translation to the geometry coordinates.
    ///
    /// \param translation A 3D/2D vector to transform the geometry.
    /// \param relative If `true`, the \p translation is directly applied to the
    /// geometry. Otherwise, the geometry center is moved to the \p translation.
    virtual GeometryBase<VectorT, MatrixT, TransformT> &Translate(
            const VectorT &translation,
            bool relative = true) = 0;
    /// \brief Apply scaling to the geometry coordinates.
    ///
    /// \param scale The scale parameter that is multiplied to the
    /// points/vertices of the geometry.
    /// \param center If `true`, the scale is applied relative to the center of
    /// the geometry. Otherwise, the scale is directly applied to the geometry,
    /// i.e. relative to the origin.
    virtual GeometryBase<VectorT, MatrixT, TransformT> &Scale(const float scale, bool center = true) = 0;
    /// \brief Apply rotation to the geometry coordinates and normals.
    ///
    /// \param R A 3D/2D vector that either defines the three angles for Euler
    /// rotation, or in the axis-angle representation the normalized vector
    /// defines the axis of rotation and the norm the angle around this axis.
    /// \param center If `true`, the rotation is applied relative to the center
    /// of the geometry. Otherwise, the rotation is directly applied to the
    /// geometry, i.e. relative to the origin.
    virtual GeometryBase<VectorT, MatrixT, TransformT> &Rotate(const MatrixT &R,
                                                               bool center = true) = 0;
};

template <typename VectorT>
class GeometryBase<VectorT, void, void> : public Geometry {
public:
    __host__ __device__ ~GeometryBase(){};  // non-virtual

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type type of object based on GeometryType.
    __host__ __device__ GeometryBase(GeometryType type, int dimension = VectorT::SizeAtCompileTime)
    : Geometry(type, dimension){};

public:
    GeometryBase<VectorT, void, void> &Clear() override = 0;
    bool IsEmpty() const override = 0;
    /// Returns min bounds for geometry coordinates.
    virtual VectorT GetMinBound() const = 0;
    /// Returns max bounds for geometry coordinates.
    virtual VectorT GetMaxBound() const = 0;
    /// Returns the center of the geometry coordinates.
    virtual VectorT GetCenter() const = 0;
};

using GeometryBase2D = GeometryBase<Eigen::Vector2f, Eigen::Matrix2f, Eigen::Matrix3f>;
using GeometryBase3D = GeometryBase<Eigen::Vector3f, Eigen::Matrix3f, Eigen::Matrix4f>;
using GeometryBaseNoTrans2D = GeometryBase<Eigen::Vector2f, void, void>;
template<int Dim>
using GeometryBaseXD = GeometryBase<Eigen::Matrix<float, Dim, 1>, Eigen::Matrix<float, Dim, Dim>, Eigen::Matrix<float, Dim + 1, Dim + 1>>;

}  // namespace geometry
}  // namespace cupoch