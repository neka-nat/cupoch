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
#include "tests/test_utility/unit_test.h"

#include <Eigen/Core>
#include <iostream>

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
void unit_test::ExpectEQ(const std::vector<uint8_t>& v0,
                         const std::vector<uint8_t>& v1) {
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
void unit_test::ExpectEQ(const std::vector<int>& v0,
                         const std::vector<int>& v1) {
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
void unit_test::ExpectEQ(const std::vector<float>& v0,
                         const std::vector<float>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    ExpectEQ(v0.data(), v1.data(), v0.size());
}
