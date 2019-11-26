#include "tests/test_utility/sort.h"

using namespace thrust;

// ----------------------------------------------------------------------------
// Greater than or Equal for sorting Eigen::Vector3d elements.
// ----------------------------------------------------------------------------
bool unit_test::sort::GE(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1) {
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
// Sort a vector of Eigen::Vector3d elements.
// ----------------------------------------------------------------------------
void unit_test::sort::Do(host_vector<Eigen::Vector3f_u>& v) {
    Eigen::Vector3f_u temp(0.0, 0.0, 0.0);
    for (size_t i = 0; i < v.size(); i++) {
        for (size_t j = 0; j < v.size(); j++) {
            if (GE(v[i], v[j])) continue;

            temp = v[j];
            v[j] = v[i];
            v[i] = temp;
        }
    }
}