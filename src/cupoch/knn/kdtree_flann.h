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

#include <thrust/host_vector.h>

#include <Eigen/Core>
#include <memory>

#include "cupoch/knn/kdtree_search_param.h"
#include "cupoch/utility/device_vector.h"

namespace flann {
template <typename T>
class Matrix;
template <typename T>
struct L2;
template <typename T>
class KDTreeCuda3dIndex;
}  // namespace flann

namespace cupoch {
namespace knn {

class KDTreeFlann {
public:
    KDTreeFlann();
    KDTreeFlann(const utility::device_vector<Eigen::Vector3f> &data);
    KDTreeFlann(const std::vector<Eigen::Vector3f> &data);
    ~KDTreeFlann();
    KDTreeFlann(const KDTreeFlann &) = delete;
    KDTreeFlann &operator=(const KDTreeFlann &) = delete;

public:
    template <typename InputIterator, int Dim>
    int Search(InputIterator first,
               InputIterator last,
               const KDTreeSearchParam &param,
               utility::device_vector<int> &indices,
               utility::device_vector<float> &distance2) const;

    template <typename InputIterator, int Dim>
    int SearchKNN(InputIterator first,
                  InputIterator last,
                  int knn,
                  utility::device_vector<int> &indices,
                  utility::device_vector<float> &distance2) const;

    template <typename InputIterator, int Dim>
    int SearchRadius(InputIterator first,
                     InputIterator last,
                     float radius,
                     int max_nn,
                     utility::device_vector<int> &indices,
                     utility::device_vector<float> &distance2) const;

    template <typename T>
    int Search(const utility::device_vector<T> &query,
               const KDTreeSearchParam &param,
               utility::device_vector<int> &indices,
               utility::device_vector<float> &distance2) const;

    template <typename T>
    int SearchKNN(const utility::device_vector<T> &query,
                  int knn,
                  utility::device_vector<int> &indices,
                  utility::device_vector<float> &distance2) const;

    template <typename T>
    int SearchRadius(const utility::device_vector<T> &query,
                     float radius,
                     int max_nn,
                     utility::device_vector<int> &indices,
                     utility::device_vector<float> &distance2) const;

    template <typename T>
    int Search(const T &query,
               const KDTreeSearchParam &param,
               thrust::host_vector<int> &indices,
               thrust::host_vector<float> &distance2) const;

    template <typename T>
    int SearchKNN(const T &query,
                  int knn,
                  thrust::host_vector<int> &indices,
                  thrust::host_vector<float> &distance2) const;

    template <typename T>
    int SearchRadius(const T &query,
                     float radius,
                     int max_nn,
                     thrust::host_vector<int> &indices,
                     thrust::host_vector<float> &distance2) const;

    template <typename T>
    int Search(const T &query,
               const KDTreeSearchParam &param,
               std::vector<int> &indices,
               std::vector<float> &distance2) const;

    template <typename T>
    int SearchKNN(const T &query,
                  int knn,
                  std::vector<int> &indices,
                  std::vector<float> &distance2) const;

    template <typename T>
    int SearchRadius(const T &query,
                     float radius,
                     int max_nn,
                     std::vector<int> &indices,
                     std::vector<float> &distance2) const;

    template <typename InputIterator, int Dim>
    bool SetRawData(InputIterator first, InputIterator last);

    template <typename T>
    bool SetRawData(const utility::device_vector<T> &data);

protected:
    utility::device_vector<float4_t> data_;
    std::unique_ptr<flann::Matrix<float>> flann_dataset_;
    std::unique_ptr<flann::KDTreeCuda3dIndex<flann::L2<float>>> flann_index_;
    size_t dimension_ = 0;
    size_t dataset_size_ = 0;
};

}  // namespace knn
}  // namespace cupoch

#include "cupoch/knn/kdtree_flann.inl"