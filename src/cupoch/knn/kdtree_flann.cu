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
#include "cupoch/knn/kdtree_flann.h"
#include "cupoch/utility/eigen.h"
#include "cupoch/utility/helper.h"

namespace cupoch {
namespace knn {

KDTreeFlann::KDTreeFlann() {}

KDTreeFlann::KDTreeFlann(const utility::device_vector<Eigen::Vector3f>& data) { SetRawData(data); }

KDTreeFlann::KDTreeFlann(const std::vector<Eigen::Vector3f> &data) {
    SetRawData(utility::device_vector<Eigen::Vector3f>(data));
}

KDTreeFlann::~KDTreeFlann() {}

template <typename T>
int KDTreeFlann::Search(const utility::device_vector<T> &query,
                        const KDTreeSearchParam &param,
                        utility::device_vector<int> &indices,
                        utility::device_vector<float> &distance2) const {
    return Search<typename utility::device_vector<T>::const_iterator,
                  T::RowsAtCompileTime>(query.begin(), query.end(), param,
                                        indices, distance2);
}

template <typename T>
int KDTreeFlann::SearchKNN(const utility::device_vector<T> &query,
                           int knn,
                           utility::device_vector<int> &indices,
                           utility::device_vector<float> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Other flann::Index::knnSearch() implementations lose performance due to
    // memory allocation/deallocation.
    if (data_.empty() || query.empty() || dataset_size_ <= 0 || knn < 0 ||
        knn > NUM_MAX_NN)
        return -1;
    T query0 = query[0];
    if (size_t(query0.size()) != dimension_) return -1;
    return SearchKNN<typename utility::device_vector<T>::const_iterator,
                     T::RowsAtCompileTime>(query.begin(), query.end(), knn,
                                           indices, distance2);
}

template <typename T>
int KDTreeFlann::SearchRadius(const utility::device_vector<T> &query,
                              float radius,
                              int max_nn,
                              utility::device_vector<int> &indices,
                              utility::device_vector<float> &distance2) const {
    // This is optimized code for heavily repeated search.
    // It is also the recommended setting for search.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory allocation/deallocation.
    if (data_.empty() || query.empty() || dataset_size_ <= 0 || max_nn < 0)
        return -1;
    T query0 = query[0];
    if (size_t(query0.size()) != dimension_) return -1;
    return SearchRadius<typename utility::device_vector<T>::const_iterator,
                        T::RowsAtCompileTime>(
            query.begin(), query.end(), radius, max_nn, indices, distance2);
}

template <typename T>
bool KDTreeFlann::SetRawData(const utility::device_vector<T> &data) {
    return SetRawData<typename utility::device_vector<T>::const_iterator,
                      T::SizeAtCompileTime>(data.begin(), data.end());
}

template <typename T>
int KDTreeFlann::Search(const T &query,
                        const KDTreeSearchParam &param,
                        thrust::host_vector<int> &indices,
                        thrust::host_vector<float> &distance2) const {
    utility::device_vector<T> query_dv(1, query);
    utility::device_vector<int> indices_dv;
    utility::device_vector<float> distance2_dv;
    auto result = Search<T>(query_dv, param, indices_dv, distance2_dv);
    indices = indices_dv;
    distance2 = distance2_dv;
    return result;
}

template <typename T>
int KDTreeFlann::SearchKNN(const T &query,
                           int knn,
                           thrust::host_vector<int> &indices,
                           thrust::host_vector<float> &distance2) const {
    utility::device_vector<T> query_dv(1, query);
    utility::device_vector<int> indices_dv;
    utility::device_vector<float> distance2_dv;
    auto result = SearchKNN<T>(query_dv, knn, indices_dv, distance2_dv);
    indices = indices_dv;
    distance2 = distance2_dv;
    return result;
}

template <typename T>
int KDTreeFlann::SearchRadius(const T &query,
                              float radius,
                              int max_nn,
                              thrust::host_vector<int> &indices,
                              thrust::host_vector<float> &distance2) const {
    utility::device_vector<T> query_dv(1, query);
    utility::device_vector<int> indices_dv;
    utility::device_vector<float> distance2_dv;
    auto result =
            SearchRadius<T>(query_dv, radius, max_nn, indices_dv, distance2_dv);
    indices = indices_dv;
    distance2 = distance2_dv;
    return result;
}

template <typename T>
int KDTreeFlann::Search(const T &query,
                        const KDTreeSearchParam &param,
                        std::vector<int> &indices,
                        std::vector<float> &distance2) const {
    utility::device_vector<T> query_dv(1, query);
    utility::device_vector<int> indices_dv;
    utility::device_vector<float> distance2_dv;
    auto result = Search<T>(query_dv, param, indices_dv, distance2_dv);
    indices.resize(indices_dv.size());
    distance2.resize(distance2_dv.size());
    copy_device_to_host(indices_dv, indices);
    copy_device_to_host(distance2_dv, distance2);
    return result;
}

template <typename T>
int KDTreeFlann::SearchKNN(const T &query,
                           int knn,
                           std::vector<int> &indices,
                           std::vector<float> &distance2) const {
    utility::device_vector<T> query_dv(1, query);
    utility::device_vector<int> indices_dv;
    utility::device_vector<float> distance2_dv;
    auto result = SearchKNN<T>(query_dv, knn, indices_dv, distance2_dv);
    indices.resize(indices_dv.size());
    distance2.resize(distance2_dv.size());
    copy_device_to_host(indices_dv, indices);
    copy_device_to_host(distance2_dv, distance2);
    return result;
}

template <typename T>
int KDTreeFlann::SearchRadius(const T &query,
                              float radius,
                              int max_nn,
                              std::vector<int> &indices,
                              std::vector<float> &distance2) const {
    utility::device_vector<T> query_dv(1, query);
    utility::device_vector<int> indices_dv;
    utility::device_vector<float> distance2_dv;
    auto result = SearchRadius<T>(query_dv, radius, max_nn, indices_dv, distance2_dv);
    indices.resize(indices_dv.size());
    distance2.resize(distance2_dv.size());
    copy_device_to_host(indices_dv, indices);
    copy_device_to_host(distance2_dv, distance2);
    return result;
}

template int KDTreeFlann::Search<Eigen::Vector3f>(
        const utility::device_vector<Eigen::Vector3f> &query,
        const KDTreeSearchParam &param,
        utility::device_vector<int> &indices,
        utility::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector3f>(
        const utility::device_vector<Eigen::Vector3f> &query,
        int knn,
        utility::device_vector<int> &indices,
        utility::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector3f>(
        const utility::device_vector<Eigen::Vector3f> &query,
        float radius,
        int max_nn,
        utility::device_vector<int> &indices,
        utility::device_vector<float> &distance2) const;
template int KDTreeFlann::Search<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        const KDTreeSearchParam &param,
        thrust::host_vector<int> &indices,
        thrust::host_vector<float> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        int knn,
        thrust::host_vector<int> &indices,
        thrust::host_vector<float> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        float radius,
        int max_nn,
        thrust::host_vector<int> &indices,
        thrust::host_vector<float> &distance2) const;
template int KDTreeFlann::Search<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<float> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        int knn,
        std::vector<int> &indices,
        std::vector<float> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        float radius,
        int max_nn,
        std::vector<int> &indices,
        std::vector<float> &distance2) const;
template bool KDTreeFlann::SetRawData<Eigen::Vector3f>(
        const utility::device_vector<Eigen::Vector3f> &data);

template int KDTreeFlann::Search<Eigen::Vector2f>(
        const utility::device_vector<Eigen::Vector2f> &query,
        const KDTreeSearchParam &param,
        utility::device_vector<int> &indices,
        utility::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector2f>(
        const utility::device_vector<Eigen::Vector2f> &query,
        int knn,
        utility::device_vector<int> &indices,
        utility::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector2f>(
        const utility::device_vector<Eigen::Vector2f> &query,
        float radius,
        int max_nn,
        utility::device_vector<int> &indices,
        utility::device_vector<float> &distance2) const;
template int KDTreeFlann::Search<Eigen::Vector2f>(
        const Eigen::Vector2f &query,
        const KDTreeSearchParam &param,
        thrust::host_vector<int> &indices,
        thrust::host_vector<float> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector2f>(
        const Eigen::Vector2f &query,
        int knn,
        thrust::host_vector<int> &indices,
        thrust::host_vector<float> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector2f>(
        const Eigen::Vector2f &query,
        float radius,
        int max_nn,
        thrust::host_vector<int> &indices,
        thrust::host_vector<float> &distance2) const;
template int KDTreeFlann::Search<Eigen::Vector2f>(
        const Eigen::Vector2f &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<float> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector2f>(
        const Eigen::Vector2f &query,
        int knn,
        std::vector<int> &indices,
        std::vector<float> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector2f>(
        const Eigen::Vector2f &query,
        float radius,
        int max_nn,
        std::vector<int> &indices,
        std::vector<float> &distance2) const;
template bool KDTreeFlann::SetRawData<Eigen::Vector2f>(
        const utility::device_vector<Eigen::Vector2f> &data);

}
}
