#pragma once

#include <thrust/host_vector.h>

#include <Eigen/Core>

#include "cupoch/utility/eigen.h"

namespace unit_test {
namespace sort {
// Greater than or Equal for sorting Eigen::Matrix<T, Dim, 1> elements.
template<typename T, int Dim>
bool GE(const Eigen::Matrix<T, Dim, 1>& v0, const Eigen::Matrix<T, Dim, 1>& v1);

// Sort a vector of Eigen::Matrix<T, Dim, 1> elements.
// method needed because std::sort failed on TravisCI/macOS (works fine on
// Linux)
template<typename T, int Dim>
void Do(thrust::host_vector<Eigen::Matrix<T, Dim, 1>>& v);
}  // namespace sort
}  // namespace unit_test