#pragma once
#include <Eigen/Core>

namespace Eigen {

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::DontAlign> MatrixXf_u;
typedef Eigen::Matrix<float, 4, 4, Eigen::DontAlign> Matrix4f_u;
typedef Eigen::Matrix<float, 3, 3, Eigen::DontAlign> Matrix3f_u;
typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> Vector3f_u;

}  // namespace Eigen