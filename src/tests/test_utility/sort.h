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

#include <Eigen/Core>

#include "cupoch/utility/eigen.h"

namespace unit_test {
namespace sort {
// Greater than or Equal for sorting Eigen::Matrix<T, Dim, 1> elements.
template <typename T, int Dim>
bool GE(const Eigen::Matrix<T, Dim, 1>& v0, const Eigen::Matrix<T, Dim, 1>& v1);

// Sort a vector of Eigen::Matrix<T, Dim, 1> elements.
// method needed because std::sort failed on TravisCI/macOS (works fine on
// Linux)
template <typename T, int Dim>
void Do(std::vector<Eigen::Matrix<T, Dim, 1>>& v);
}  // namespace sort
}  // namespace unit_test