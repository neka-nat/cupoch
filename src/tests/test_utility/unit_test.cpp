#include <Eigen/Core>
#include <iostream>

#include "tests/test_utility/unit_test.h"

using namespace thrust;
using namespace std;
using namespace unit_test;

// ----------------------------------------------------------------------------
// Default message to use for tests missing an implementation.
// ----------------------------------------------------------------------------
void unit_test::NotImplemented() {
    cout << "\033[0;32m"
         << "[          ] "
         << "\033[0;0m";
    cout << "\033[0;31m"
         << "Not implemented."
         << "\033[0;0m" << endl;

    GTEST_NONFATAL_FAILURE_("Not implemented");
}

// ----------------------------------------------------------------------------
// Test equality of two arrays of uint8_t.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const uint8_t* const v0,
                         const uint8_t* const v1,
                         const size_t& size) {
    for (size_t i = 0; i < size; i++) EXPECT_EQ(v0[i], v1[i]);
}

// ----------------------------------------------------------------------------
// Test equality of two vectors of uint8_t.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const host_vector<uint8_t>& v0, const host_vector<uint8_t>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    ExpectEQ(v0.data(), v1.data(), v0.size());
}

// ----------------------------------------------------------------------------
// Test equality of two arrays of int.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const int* const v0,
                         const int* const v1,
                         const size_t& size) {
    for (size_t i = 0; i < size; i++) EXPECT_EQ(v0[i], v1[i]);
}

// ----------------------------------------------------------------------------
// Test equality of two vectors of int.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const host_vector<int>& v0, const host_vector<int>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    ExpectEQ(v0.data(), v1.data(), v0.size());
}

// ----------------------------------------------------------------------------
// Test equality of two arrays of float.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const float* const v0,
                         const float* const v1,
                         const size_t& size) {
    for (size_t i = 0; i < size; i++) EXPECT_NEAR(v0[i], v1[i], THRESHOLD_1E_4);
}

// ----------------------------------------------------------------------------
// Test equality of two vectors of float.
// ----------------------------------------------------------------------------
void unit_test::ExpectEQ(const host_vector<float>& v0, const host_vector<float>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    ExpectEQ(v0.data(), v1.data(), v0.size());
}
