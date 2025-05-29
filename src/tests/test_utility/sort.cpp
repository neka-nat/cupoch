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
#include "tests/test_utility/sort.h"

using namespace thrust;

// ----------------------------------------------------------------------------
// Greater than or Equal for sorting Eigen::Matrix<T, Dim, 1> elements.
// ----------------------------------------------------------------------------
template <typename T, int Dim>
bool unit_test::sort::GE(const Eigen::Matrix<T, Dim, 1>& v0,
                         const Eigen::Matrix<T, Dim, 1>& v1) {
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
template <typename T, int Dim>
void unit_test::sort::Do(std::vector<Eigen::Matrix<T, Dim, 1>>& v) {
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

template bool unit_test::sort::GE<float, 3>(const Eigen::Vector3f& v0,
                                            const Eigen::Vector3f& v1);
template bool unit_test::sort::GE<int, 3>(const Eigen::Vector3i& v0,
                                          const Eigen::Vector3i& v1);
template void unit_test::sort::Do<float, 3>(std::vector<Eigen::Vector3f>& v);
template void unit_test::sort::Do<int, 3>(std::vector<Eigen::Vector3i>& v);
