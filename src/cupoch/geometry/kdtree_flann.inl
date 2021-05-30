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
#include "cupoch/geometry/kdtree_flann.h"
#define FLANN_USE_CUDA
#include <flann/flann.hpp>
#undef FLANN_USE_CUDA

#include "cupoch/utility/console.h"

namespace cupoch {
namespace geometry {

template <int Dim>
struct convert_float4_functor {
    __device__ float4_t
    operator()(const Eigen::Matrix<float, Dim, 1> &x) const {
        float4_t res = {0};
        float *pt = (float *)&res;
        constexpr int num = (Dim > 4) ? 4 : Dim;
#pragma unroll
        for (int i = 0; i < num; i++) {
            pt[i] = x[i];
        }
        return res;
    }
};

template <typename InputIterator, int Dim>
int KDTreeFlann::Search(InputIterator first,
                        InputIterator last,
                        const KDTreeSearchParam &param,
                        utility::device_vector<int> &indices,
                        utility::device_vector<float> &distance2) const {
    switch (param.GetSearchType()) {
        case KDTreeSearchParam::SearchType::Knn:
            return SearchKNN<InputIterator, Dim>(
                    first, last, ((const KDTreeSearchParamKNN &)param).knn_,
                    indices, distance2);
        case KDTreeSearchParam::SearchType::Radius:
            return SearchRadius<InputIterator, Dim>(
                    first, last,
                    ((const KDTreeSearchParamRadius &)param).radius_,
                    ((const KDTreeSearchParamRadius &)param).max_nn_, indices,
                    distance2);
        default:
            return -1;
    }
    return -1;
}

template <typename InputIterator, int Dim>
int KDTreeFlann::SearchKNN(InputIterator first,
                           InputIterator last,
                           int knn,
                           utility::device_vector<int> &indices,
                           utility::device_vector<float> &distance2) const {
    convert_float4_functor<Dim> func;
    size_t num_query = thrust::distance(first, last);
    utility::device_vector<float4_t> query_f4(num_query);
    thrust::transform(first, last, query_f4.begin(), func);
    flann::Matrix<float> query_flann(
            (float *)(thrust::raw_pointer_cast(query_f4.data())), num_query,
            dimension_, sizeof(float) * 4);
    const int total_size = num_query * knn;
    indices.resize(total_size);
    distance2.resize(total_size);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices.data()),
                                     query_flann.rows, knn);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2.data()),
                                     query_flann.rows, knn);
    flann::SearchParams param;
    param.matrices_in_gpu_ram = true;
    int k = flann_index_->knnSearch(query_flann, indices_flann, dists_flann,
                                    knn, param);
    return k;
}

template <typename InputIterator, int Dim>
int KDTreeFlann::SearchRadius(InputIterator first,
                              InputIterator last,
                              float radius,
                              int max_nn,
                              utility::device_vector<int> &indices,
                              utility::device_vector<float> &distance2) const {
    convert_float4_functor<Dim> func;
    size_t num_query = thrust::distance(first, last);
    utility::device_vector<float4_t> query_f4(num_query);
    thrust::transform(first, last, query_f4.begin(), func);
    flann::Matrix<float> query_flann(
            (float *)(thrust::raw_pointer_cast(query_f4.data())), num_query,
            dimension_, sizeof(float) * 4);
    flann::SearchParams param(-1, 0.0);
    param.max_neighbors = max_nn;
    param.matrices_in_gpu_ram = true;
    indices.resize(num_query * max_nn);
    distance2.resize(num_query * max_nn);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices.data()),
                                     query_flann.rows, max_nn);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2.data()),
                                     query_flann.rows, max_nn);
    int k = flann_index_->radiusSearch(query_flann, indices_flann, dists_flann,
                                       float(radius * radius), param);
    return k;
}

template <typename InputIterator, int Dim>
bool KDTreeFlann::SetRawData(InputIterator first, InputIterator last) {
    dimension_ = Dim;
    dataset_size_ = thrust::distance(first, last);
    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning(
                "[KDTreeFlann::SetRawData] Failed due to no data.\n");
        return false;
    }
    data_.resize(dataset_size_);
    convert_float4_functor<Dim> func;
    thrust::transform(first, last, data_.begin(), func);
    flann_dataset_.reset(new flann::Matrix<float>(
            (float *)thrust::raw_pointer_cast(data_.data()), dataset_size_,
            dimension_, sizeof(float) * 4));
    flann::KDTreeCuda3dIndexParams index_params;
    flann_index_.reset(new flann::KDTreeCuda3dIndex<flann::L2<float>>(
            *flann_dataset_, index_params));
    flann_index_->buildIndex();
    return true;
}

}  // namespace geometry
}  // namespace cupoch