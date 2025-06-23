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
#include "cupoch/geometry/geometry.h"
#include "cupoch/utility/device_vector.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace geometry {

const utility::device_vector<Eigen::Vector3f>& ConvertVector3fVectorRef(const Geometry &geometry);
const std::vector<Eigen::Vector3f> ConvertVector3fStdVector(const Geometry &geometry);

/// Get Rotation Matrix from XYZ RotationType.
Eigen::Matrix3f GetRotationMatrixFromXYZ(const Eigen::Vector3f &rotation);
/// Get Rotation Matrix from YZX RotationType.
Eigen::Matrix3f GetRotationMatrixFromYZX(const Eigen::Vector3f &rotation);
/// Get Rotation Matrix from ZXY RotationType.
Eigen::Matrix3f GetRotationMatrixFromZXY(const Eigen::Vector3f &rotation);
/// Get Rotation Matrix from XZY RotationType.
Eigen::Matrix3f GetRotationMatrixFromXZY(const Eigen::Vector3f &rotation);
/// Get Rotation Matrix from ZYX RotationType.
Eigen::Matrix3f GetRotationMatrixFromZYX(const Eigen::Vector3f &rotation);
/// Get Rotation Matrix from YXZ RotationType.
Eigen::Matrix3f GetRotationMatrixFromYXZ(const Eigen::Vector3f &rotation);
/// Get Rotation Matrix from AxisAngle RotationType.
Eigen::Matrix3f GetRotationMatrixFromAxisAngle(const Eigen::Vector3f &rotation);
/// Get Rotation Matrix from Quaternion.
Eigen::Matrix3f GetRotationMatrixFromQuaternion(
        const Eigen::Vector4f &rotation);

void ResizeAndPaintUniformColor(utility::device_vector<Eigen::Vector3f> &colors,
                                const size_t size,
                                const Eigen::Vector3f &color);

/// \brief Transforms all points with the transformation matrix.
///
/// \param transformation 4x4 matrix for transformation.
/// \param points A list of points to be transformed.
template <int Dim>
void TransformPoints(
        const Eigen::Matrix<float, Dim + 1, Dim + 1> &transformation,
        utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points);
template <int Dim>
void TransformPoints(
        cudaStream_t stream,
        const Eigen::Matrix<float, Dim + 1, Dim + 1> &transformation,
        utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points);
/// \brief Transforms the normals with the transformation matrix.
///
/// \param transformation 4x4 matrix for transformation.
/// \param normals A list of normals to be transformed.
void TransformNormals(const Eigen::Matrix4f &transformation,
                      utility::device_vector<Eigen::Vector3f> &normals);
void TransformNormals(cudaStream_t stream,
                      const Eigen::Matrix4f &transformation,
                      utility::device_vector<Eigen::Vector3f> &normals);

/// \brief Transforms all covariance matrices with the transformation.
///
/// \param transformation 4x4 matrix for transformation.
/// \param covariances A list of covariance matrices to be transformed.
void TransformCovariances(cudaStream_t stream,
                          const Eigen::Matrix4f& transformation,
                          utility::device_vector<Eigen::Matrix3f>& covariances);
void TransformCovariances(const Eigen::Matrix4f& transformation,
                          utility::device_vector<Eigen::Matrix3f>& covariances);

/// \brief Apply translation to the geometry coordinates.
///
/// \param translation A 3D vector to transform the geometry.
/// \param points A list of points to be transformed.
/// \param relative If `true`, the \p translation is directly applied to the
/// \points. Otherwise, the center of the \points is moved to the \p
/// translation.
template <int Dim>
void TranslatePoints(
        const Eigen::Matrix<float, Dim, 1> &translation,
        utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
        bool relative);
/// \brief Scale the coordinates of all points by the scaling factor \p
/// scale.
///
/// \param scale If `true`, the scale is applied relative to the center of
/// the geometry. Otherwise, the scale is directly applied to the geometry,
/// i.e. relative to the origin. \param points A list of points to be
/// transformed. \param center If `true`, then the scale is applied to the
/// centered geometry.
template <int Dim>
void ScalePoints(const float scale,
                 utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
                 bool center);
/// \brief Rotate all points with the rotation matrix \p R.
///
/// \param R A 3D vector that either defines the three angles for Euler
/// rotation, or in the axis-angle representation the normalized vector
/// defines the axis of rotation and the norm the angle around this axis.
/// \param points A list of points to be transformed.
/// \param center If `true`, the rotation is applied relative to the center
/// of the geometry. Otherwise, the rotation is directly applied to the
/// geometry, i.e. relative to the origin.
template <int Dim>
void RotatePoints(const Eigen::Matrix<float, Dim, Dim> &R,
                  utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
                  bool center);
template <int Dim>
void RotatePoints(cudaStream_t stream,
                  const Eigen::Matrix<float, Dim, Dim> &R,
                  utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
                  bool center);
/// \brief Rotate all normals with the rotation matrix \p R.
///
/// \param R A 3D vector that either defines the three angles for Euler
/// rotation, or in the axis-angle representation the normalized vector
/// defines the axis of rotation and the norm the angle around this axis.
/// \param normals A list of normals to be transformed.
void RotateNormals(const Eigen::Matrix3f &R,
                   utility::device_vector<Eigen::Vector3f> &normals);
void RotateNormals(cudaStream_t stream,
                   const Eigen::Matrix3f &R,
                   utility::device_vector<Eigen::Vector3f> &normals);

/// \brief Rotate all covariance matrices with the rotation matrix \p R.
///
/// \param R A 3x3 rotation matrix
/// \param covariances A list of covariance matrices to be transformed.
void RotateCovariances(const Eigen::Matrix3f& R,
                       utility::device_vector<Eigen::Matrix3f>& covariances);
void RotateCovariances(cudaStream_t stream,
                       const Eigen::Matrix3f& R,
                       utility::device_vector<Eigen::Matrix3f>& covariances);

}  // namespace geometry
}  // namespace cupoch