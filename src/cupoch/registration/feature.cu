/**
 * Copyright (c) 2021 Neka-Nat
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
#include "cupoch/registration/feature.h"

namespace cupoch {
namespace registration {

template <int Dim>
Feature<Dim>::Feature(){};

template <int Dim>
Feature<Dim>::Feature(const Feature<Dim> &other) : data_(other.data_) {}

template <int Dim>
Feature<Dim>::~Feature() {}

template <int Dim>
void Feature<Dim>::Resize(int n) {
    data_.resize(n);
}

template <int Dim>
size_t Feature<Dim>::Dimension() const {
    return Dim;
}

template <int Dim>
size_t Feature<Dim>::Num() const {
    return data_.size();
}

template <int Dim>
bool Feature<Dim>::IsEmpty() const {
    return data_.empty();
}

template <int Dim>
thrust::host_vector<Eigen::Matrix<float, Dim, 1>> Feature<Dim>::GetData()
        const {
    thrust::host_vector<Eigen::Matrix<float, Dim, 1>> h_data = data_;
    return h_data;
}

template <int Dim>
void Feature<Dim>::SetData(
        const thrust::host_vector<Eigen::Matrix<float, Dim, 1>> &data) {
    data_ = data;
}

template class Feature<33>;

}
}