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
#pragma once

// TEST_DATA_DIR defined in CMakeLists.txt
// Put it here to avoid editor warnings
#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR
#endif

#include <gtest/gtest.h>

#include <Eigen/Core>

#include "tests/test_utility/print.h"
#include "tests/test_utility/rand.h"
#include "tests/test_utility/sort.h"

namespace unit_test {
// thresholds for comparing floating point values
const float THRESHOLD_1E_4 = 1e-4;

// Eigen Zero()
const Eigen::Vector2f Zero2f = Eigen::Vector2f::Zero();
const Eigen::Vector3f Zero3f = Eigen::Vector3f::Zero();
const Eigen::Matrix<float, 6, 1> Zero6f = Eigen::Matrix<float, 6, 1>::Zero();
const Eigen::Vector2i Zero2i = Eigen::Vector2i::Zero();

// Mechanism for reporting unit tests for which there is no implementation yet.
void NotImplemented();

// Equal test.
template <class T, int M, int N, int A>
void ExpectEQ(const Eigen::Matrix<T, M, N, A>& v0,
              const Eigen::Matrix<T, M, N, A>& v1,
              double threshold = THRESHOLD_1E_4) {
    EXPECT_EQ(v0.size(), v1.size());
    for (int i = 0; i < v0.size(); i++)
        EXPECT_NEAR(v0.coeff(i), v1.coeff(i), threshold);
}
template <class T, int M, int N, int A>
void ExpectEQ(const std::vector<Eigen::Matrix<T, M, N, A>>& v0,
              const std::vector<Eigen::Matrix<T, M, N, A>>& v1,
              double threshold = THRESHOLD_1E_4) {
    EXPECT_EQ(v0.size(), v1.size());
    for (size_t i = 0; i < v0.size(); i++) ExpectEQ(v0[i], v1[i], threshold);
}
template <class T, int M, int N, int A>
void ExpectEQ(
        const std::vector<Eigen::Matrix<T, M, N, A>,
                          Eigen::aligned_allocator<Eigen::Matrix<T, M, N, A>>>&
                v0,
        const std::vector<Eigen::Matrix<T, M, N, A>,
                          Eigen::aligned_allocator<Eigen::Matrix<T, M, N, A>>>&
                v1,
        double threshold = THRESHOLD_1E_4) {
    EXPECT_EQ(v0.size(), v1.size());
    for (size_t i = 0; i < v0.size(); i++) ExpectEQ(v0[i], v1[i], threshold);
}

// Less than or Equal test.
template <class T, int M, int N, int A>
void ExpectLE(const Eigen::Matrix<T, M, N, A>& v0,
              const Eigen::Matrix<T, M, N, A>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    for (int i = 0; i < v0.size(); i++) EXPECT_LE(v0.coeff(i), v1.coeff(i));
}
template <class T, int M, int N, int A>
void ExpectLE(const Eigen::Matrix<T, M, N, A>& v0,
              const std::vector<Eigen::Matrix<T, M, N, A>>& v1) {
    for (size_t i = 0; i < v1.size(); i++) ExpectLE(v0, v1[i]);
}
template <class T, int M, int N, int A>
void ExpectLE(const std::vector<Eigen::Matrix<T, M, N, A>>& v0,
              const std::vector<Eigen::Matrix<T, M, N, A>>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    for (size_t i = 0; i < v0.size(); i++) ExpectLE(v0[i], v1[i]);
}

// Greater than or Equal test.
template <class T, int M, int N, int A>
void ExpectGE(const Eigen::Matrix<T, M, N, A>& v0,
              const Eigen::Matrix<T, M, N, A>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    for (int i = 0; i < v0.size(); i++) EXPECT_GE(v0.coeff(i), v1.coeff(i));
}
template <class T, int M, int N, int A>
void ExpectGE(const Eigen::Matrix<T, M, N, A>& v0,
              const std::vector<Eigen::Matrix<T, M, N, A>>& v1) {
    for (size_t i = 0; i < v1.size(); i++) ExpectGE(v0, v1[i]);
}
template <class T, int M, int N, int A>
void ExpectGE(const std::vector<Eigen::Matrix<T, M, N, A>>& v0,
              const std::vector<Eigen::Matrix<T, M, N, A>>& v1) {
    EXPECT_EQ(v0.size(), v1.size());
    for (size_t i = 0; i < v0.size(); i++) ExpectGE(v0[i], v1[i]);
}

// Test equality of two arrays of uint8_t.
void ExpectEQ(const uint8_t* const v0,
              const uint8_t* const v1,
              const size_t& size);

// Test equality of two vectors of uint8_t.
void ExpectEQ(const std::vector<uint8_t>& v0,
              const std::vector<uint8_t>& v1);

// Test equality of two arrays of int.
void ExpectEQ(const int* const v0, const int* const v1, const size_t& size);

// Test equality of two vectors of int.
void ExpectEQ(const std::vector<int>& v0,
              const std::vector<int>& v1);

// Test equality of two arrays of float.
void ExpectEQ(const float* const v0, const float* const v1, const size_t& size);

// Test equality of two vectors of float.
void ExpectEQ(const std::vector<float>& v0,
              const std::vector<float>& v1);

// Test equality of two arrays of double.
void ExpectEQ(const float* const v0, const float* const v1, const size_t& size);

// Reinterpret cast from uint8_t* to float*.
template <class T>
T* const Cast(uint8_t* data) {
    return reinterpret_cast<T* const>(data);
}
}  // namespace unit_test