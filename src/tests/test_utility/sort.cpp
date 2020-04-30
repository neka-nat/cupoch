#include "tests/test_utility/sort.h"

using namespace thrust;

// ----------------------------------------------------------------------------
// Greater than or Equal for sorting Eigen::Matrix<T, Dim, 1> elements.
// ----------------------------------------------------------------------------
template<typename T, int Dim>
bool unit_test::sort::GE(const Eigen::Matrix<T, Dim, 1>& v0, const Eigen::Matrix<T, Dim, 1>& v1) {
    if (v0(0, 0) > v1(0, 0)) return true;

    if (v0(0, 0) == v1(0, 0)) {
        if (v0(1, 0) > v1(1, 0)) return true;

        if (v0(1, 0) == v1(1, 0)) {
            if (v0(2, 0) >= v1(2, 0)) return true;
        }
    }

    return false;
}

// ----------------------------------------------------------------------------
// Sort a vector of Eigen::Matrix elements.
// ----------------------------------------------------------------------------
template<typename T, int Dim>
void unit_test::sort::Do(host_vector<Eigen::Matrix<T, Dim, 1>>& v) {
    Eigen::Matrix<T, Dim, 1> temp = Eigen::Matrix<T, Dim, 1>::Zero();
    for (size_t i = 0; i < v.size(); i++) {
        for (size_t j = 0; j < v.size(); j++) {
            if (GE(v[i], v[j])) continue;

            temp = v[j];
            v[j] = v[i];
            v[i] = temp;
        }
    }
}

template bool unit_test::sort::GE<float, 3>(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1);
template bool unit_test::sort::GE<int, 3>(const Eigen::Vector3i& v0, const Eigen::Vector3i& v1);
template void unit_test::sort::Do<float, 3>(host_vector<Eigen::Vector3f>& v);
template void unit_test::sort::Do<int, 3>(host_vector<Eigen::Vector3i>& v);