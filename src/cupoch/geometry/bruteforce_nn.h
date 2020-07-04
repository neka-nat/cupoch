#pragma once

#include <Eigen/Core>

#include "cupoch/utility/device_vector.h"

namespace cupoch {
namespace geometry {

template <int Dim>
void BruteForceNN(
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>>& ref,
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>>& query,
        utility::device_vector<int>& indices,
        utility::device_vector<float>& distances);

}
}  // namespace cupoch

#include "cupoch/geometry/bruteforce_nn.inl"