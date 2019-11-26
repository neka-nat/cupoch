#pragma once

#include <Eigen/Core>
#include "cupoc/utility/eigen.h"
#include <thrust/host_vector.h>

namespace unit_test {
namespace sort {
// Greater than or Equal for sorting Eigen::Vector3f elements.
bool GE(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1);

// Sort a vector of Eigen::Vector3f elements.
// method needed because std::sort failed on TravisCI/macOS (works fine on
// Linux)
void Do(thrust::host_vector<Eigen::Vector3f_u>& v);
}  // namespace Sort
}  // namespace unit_test