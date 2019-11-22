#pragma once
#include <Eigen/Core>

namespace Eigen {

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::DontAlign> MatrixXf_u;
typedef Eigen::Matrix<float, 4, 4, Eigen::DontAlign> Matrix4f_u;
typedef Eigen::Matrix<float, 3, 3, Eigen::DontAlign> Matrix3f_u;
typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> Vector3f_u;

__host__ __device__
inline bool operator<(const Eigen::Vector3i &lhs, const Eigen::Vector3i &rhs) {
    if (lhs[0] != rhs[0]) return lhs[0] < rhs[0];
    if (lhs[1] != rhs[1]) return lhs[1] < rhs[1];
    if (lhs[2] != rhs[2]) return lhs[2] < rhs[2];
    return false;
}

}  // namespace Eigen