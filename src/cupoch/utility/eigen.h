#pragma once
#include <Eigen/Core>
#include <thrust/functional.h>
#include <thrust/tuple.h>

namespace Eigen {

/// Extending Eigen namespace by adding frequently used matrix type
typedef Eigen::Matrix<float, 6, 6> Matrix6f;
typedef Eigen::Matrix<float, 6, 1> Vector6f;

typedef Eigen::Matrix<float, 4, 4, Eigen::DontAlign> Matrix4f_u;
typedef Eigen::Matrix<float, 6, 6, Eigen::DontAlign> Matrix6f_u;

}  // namespace Eigen

namespace cupoch {
namespace utility {

template <typename VecType>
struct jacobian_residual_functor {
    __device__
    virtual void operator() (int i, VecType& vec, float& r) const = 0;
};

/// Function to transform 6D motion vector to 4D motion matrix
/// Reference:
/// https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html#TutorialGeoTransform
Eigen::Matrix4f TransformVector6fToMatrix4f(const Eigen::Vector6f &input);

/// Function to solve Ax=b
template<int Dim>
thrust::tuple<bool, Eigen::Matrix<float, Dim, 1>> SolveLinearSystemPSD(
        const Eigen::Matrix<float, Dim, Dim> &A,
        const Eigen::Matrix<float, Dim, 1> &b,
        bool check_symmetric = false,
        bool check_det = false);

/// Function to solve Jacobian system
/// Input: 6x6 Jacobian matrix and 6-dim residual vector.
/// Output: tuple of is_success, 4x4 extrinsic matrices.
thrust::tuple<bool, Eigen::Matrix4f> SolveJacobianSystemAndObtainExtrinsicMatrix(
        const Eigen::Matrix6f &JTJ, const Eigen::Vector6f &JTr);

/// Function to compute JTJ and Jtr
/// Input: function pointer f and total number of rows of Jacobian matrix
/// Output: JTJ, JTr, sum of r^2
/// Note: f takes index of row, and outputs corresponding residual and row
/// vector.
template <typename MatType, typename VecType>
thrust::tuple<MatType, VecType, float> ComputeJTJandJTr(
        jacobian_residual_functor<VecType>& f,
        int iteration_num,
        bool verbose = true);

}  // namespace utility
}  // namespace cupoch