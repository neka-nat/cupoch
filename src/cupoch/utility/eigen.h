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
#include <thrust/tuple.h>

#include <Eigen/Core>

namespace Eigen {

/// Extending Eigen namespace by adding frequently used matrix type
typedef Eigen::Matrix<float, 6, 6> Matrix6f;
typedef Eigen::Matrix<float, 6, 1> Vector6f;

typedef Eigen::Matrix<float, 4, 4, Eigen::DontAlign> Matrix4f_u;
typedef Eigen::Matrix<float, 6, 6, Eigen::DontAlign> Matrix6f_u;

typedef Eigen::Matrix<unsigned short, 3, 1> Vector3ui16;

}  // namespace Eigen

namespace cupoch {
namespace utility {

template <typename VecType>
struct jacobian_residual_functor {
    __device__ virtual void operator()(int i, VecType &vec, float &r) const = 0;
};

template <typename VecType, int NumJ>
struct multiple_jacobians_residuals_functor {
    __device__ virtual void operator()(int i,
                                       VecType J_r[NumJ],
                                       float r[NumJ]) const = 0;
};

/// Function to transform 6D motion vector to 4D motion matrix
Eigen::Matrix4f TransformVector6fToMatrix4f(const Eigen::Vector6f &input);
/// Function to transform 4D motion matrix to 6D motion vector
Eigen::Vector6f TransformMatrix4fToVector6f(const Eigen::Matrix4f &input);

/// Function to solve Ax=b
template <int Dim>
thrust::tuple<bool, Eigen::Matrix<float, Dim, 1>> SolveLinearSystemPSD(
        const Eigen::Matrix<float, Dim, Dim> &A,
        const Eigen::Matrix<float, Dim, 1> &b,
        bool check_symmetric = false,
        bool check_det = false);

/// Function to solve Jacobian system
/// Input: 6x6 Jacobian matrix and 6-dim residual vector.
/// Output: tuple of is_success, 4x4 extrinsic matrices.
thrust::tuple<bool, Eigen::Matrix4f>
SolveJacobianSystemAndObtainExtrinsicMatrix(const Eigen::Matrix6f &JTJ,
                                            const Eigen::Vector6f &JTr);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: f takes index of row, and outputs corresponding residual and row
/// vector.
template <typename MatType, typename VecType, typename FuncType>
thrust::tuple<MatType, VecType, float> ComputeJTJandJTr(const FuncType &f,
                                                        int iteration_num,
                                                        bool verbose = true);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: f takes index of row, and outputs corresponding residual and row
/// vector.
template <typename MatType, typename VecType, int NumJ, typename FuncType>
thrust::tuple<MatType, VecType, float> ComputeJTJandJTr(const FuncType &f,
                                                        int iteration_num,
                                                        bool verbose = true);

template <typename MatType,
          typename VecType,
          int NumJ,
          typename FuncJType,
          typename FuncW1Type,
          typename FuncW2Type>
thrust::tuple<MatType, VecType, float, float> ComputeWeightedJTJandJTr(
        const FuncJType &fj,
        const FuncW1Type &fw_reduce,
        const FuncW2Type &fw_trans,
        const int iteration_num,
        bool verbose = true);

Eigen::Matrix3f RotationMatrixX(float radians);
Eigen::Matrix3f RotationMatrixY(float radians);
Eigen::Matrix3f RotationMatrixZ(float radians);

}  // namespace utility
}  // namespace cupoch

#include "cupoch/utility/eigen.inl"