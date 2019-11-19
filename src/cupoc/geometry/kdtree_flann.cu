
#include "cupoc/geometry/kdtree_flann.h"
#define FLANN_USE_CUDA
#include <flann/flann.hpp>
#include "cupoc/utility/console.h"

using namespace cupoc;
using namespace cupoc::geometry;

namespace {

template<typename ElementType, typename ArrayType>
struct indexed_copy_functor {
    indexed_copy_functor(int index)
        : index_(index) {};
    const int index_;
    __device__
    ArrayType operator()(const ElementType& x, const ArrayType& idxs) {
        ArrayType ans = idxs;
        ans[index_] = x;
        return ans;
    }
};

void ConvertIndicesAndDistaces(int knn, int n_query, thrust::device_vector<int> indices_dv,
                               thrust::device_vector<float> distance2_dv,
                               thrust::device_vector<KNNIndices> &indices,
                               thrust::device_vector<KNNDistances> &distance2) {
    const int total_size = knn * n_query;
    indices.resize(n_query);
    distance2.resize(n_query);
    thrust::for_each(indices.begin(), indices.end(), [] __device__ (KNNIndices& idxs){idxs.fill(-1);});
    thrust::for_each(distance2.begin(), distance2.end(), [] __device__ (KNNDistances& dist){dist.fill(-1.0);});
    for (int k = 0; k < knn; ++k) {
        thrust::strided_range<thrust::device_vector<int>::iterator> itr_idxs(indices_dv.begin() + k, indices_dv.begin() + total_size + 1 - knn + k, knn);
        indexed_copy_functor<int, KNNIndices> icf_i(k);
        thrust::transform(itr_idxs.begin(), itr_idxs.end(), indices.begin(), indices.begin(), icf_i);
        thrust::strided_range<thrust::device_vector<float>::iterator> itr_dist(distance2_dv.begin() + k, distance2_dv.begin() + total_size + 1 - knn + k, knn);
        indexed_copy_functor<float, KNNDistances> icf_d(k);
        thrust::transform(itr_dist.begin(), itr_dist.end(), distance2.begin(), distance2.begin(), icf_d);
    }
}

}


KDTreeFlann::KDTreeFlann() {}

KDTreeFlann::KDTreeFlann(const Eigen::MatrixXf &data) {SetMatrixData(data);}
KDTreeFlann::KDTreeFlann(const PointCloud &data) {SetGeometry(data);}

KDTreeFlann::~KDTreeFlann() {}

bool KDTreeFlann::SetMatrixData(const Eigen::MatrixXf_u &data) {
    return SetRawData(Eigen::Map<const Eigen::MatrixXf_u>(
            data.data(), data.rows(), data.cols()));
}

bool KDTreeFlann::SetGeometry(const PointCloud &geometry) {
    return SetRawData(Eigen::Map<const Eigen::MatrixXf_u>(
        (const float *)(thrust::raw_pointer_cast(geometry.points_.data())), 3, geometry.points_.size()));
}

template <typename T>
int KDTreeFlann::Search(const thrust::device_vector<T> &query,
                        const KDTreeSearchParam &param,
                        thrust::device_vector<KNNIndices> &indices,
                        thrust::device_vector<KNNDistances> &distance2) const {
    switch (param.GetSearchType()) {
        case KDTreeSearchParam::SearchType::Knn:
            return SearchKNN(query, ((const KDTreeSearchParamKNN &)param).knn_,
                             indices, distance2);
        case KDTreeSearchParam::SearchType::Radius:
            return SearchRadius(
                    query, ((const KDTreeSearchParamRadius &)param).radius_,
                    indices, distance2);
        case KDTreeSearchParam::SearchType::Hybrid:
            return SearchHybrid(
                    query, ((const KDTreeSearchParamHybrid &)param).radius_,
                    ((const KDTreeSearchParamHybrid &)param).max_nn_, indices,
                    distance2);
        default:
            return -1;
    }
    return -1;
}

template <typename T>
int KDTreeFlann::SearchKNN(const thrust::device_vector<T> &query,
                           int knn,
                           thrust::device_vector<KNNIndices> &indices,
                           thrust::device_vector<KNNDistances> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Other flann::Index::knnSearch() implementations lose performance due to
    // memory allocation/deallocation.
    if (data_.empty() || query.empty() || dataset_size_ <= 0 || knn < 0 || knn > NUM_MAX_NN) return -1;
    T query0 = query[0];
    if (size_t(query0.size()) != dimension_) return -1;
    flann::Matrix<float> query_flann((float *)(thrust::raw_pointer_cast(query.data())), query.size(), dimension_);
    const int total_size = query.size() * knn;
    thrust::device_vector<int> indices_dv(total_size);
    thrust::device_vector<float> distance2_dv(total_size);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices_dv.data()), query_flann.rows, knn);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2_dv.data()), query_flann.rows, knn);
    int k = flann_index_->knnSearch(query_flann, indices_flann, dists_flann,
                                    knn, flann::SearchParams(-1, 0.0));
    ConvertIndicesAndDistaces(knn, query.size(), indices_dv, distance2_dv, indices, distance2);
    return k;
}

template <typename T>
int KDTreeFlann::SearchRadius(const thrust::device_vector<T> &query,
                              float radius,
                              thrust::device_vector<KNNIndices> &indices,
                              thrust::device_vector<KNNDistances> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Since max_nn is not given, we let flann to do its own memory management.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory management and CPU caching.
    if (data_.empty() || query.empty() || dataset_size_ <= 0) return -1;
    T query0 = query[0];
    if (size_t(query0.size()) != dimension_) return -1;
    flann::Matrix<float> query_flann((float *)(thrust::raw_pointer_cast(query.data())), query.size(), dimension_);
    flann::SearchParams param(-1, 0.0);
    param.max_neighbors = NUM_MAX_NN;
    thrust::device_vector<int> indices_dv(query.size() * NUM_MAX_NN);
    thrust::device_vector<float> distance2_dv(query.size() * NUM_MAX_NN);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices_dv.data()), query_flann.rows, NUM_MAX_NN);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2_dv.data()), query_flann.rows, NUM_MAX_NN);
    int k = flann_index_->radiusSearch(query_flann, indices_flann, dists_flann,
                                       float(radius * radius), param);
    ConvertIndicesAndDistaces(NUM_MAX_NN, query.size(), indices_dv, distance2_dv, indices, distance2);
    return k;
}

template <typename T>
int KDTreeFlann::SearchHybrid(const thrust::device_vector<T> &query,
                              float radius,
                              int max_nn,
                              thrust::device_vector<KNNIndices> &indices,
                              thrust::device_vector<KNNDistances> &distance2) const {
    // This is optimized code for heavily repeated search.
    // It is also the recommended setting for search.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory allocation/deallocation.
    if (data_.empty() || query.empty() || dataset_size_ <= 0 || max_nn < 0 || max_nn > NUM_MAX_NN) return -1;
    T query0 = query[0];
    if (size_t(query0.size()) != dimension_) return -1;
    flann::Matrix<float> query_flann((float *)(thrust::raw_pointer_cast(query.data())), query.size(), dimension_);
    flann::SearchParams param(-1, 0.0);
    param.max_neighbors = max_nn;
    thrust::device_vector<int> indices_dv(query.size() * max_nn);
    thrust::device_vector<float> distance2_dv(query.size() * max_nn);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices_dv.data()), query_flann.rows, max_nn);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2_dv.data()), query_flann.rows, max_nn);
    int k = flann_index_->radiusSearch(query_flann, indices_flann, dists_flann,
                                       float(radius * radius), param);
    ConvertIndicesAndDistaces(max_nn, query.size(), indices_dv, distance2_dv, indices, distance2);
    return k;
}

bool KDTreeFlann::SetRawData(const Eigen::Map<const Eigen::MatrixXf_u> &data) {
    dimension_ = data.rows();
    dataset_size_ = data.cols();
    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning(
                "[KDTreeFlann::SetRawData] Failed due to no data.\n");
        return false;
    }
    data_.resize(dataset_size_ * dimension_);
    memcpy(thrust::raw_pointer_cast(data_.data()), data.data(),
           dataset_size_ * dimension_ * sizeof(float));
    flann_dataset_.reset(new flann::Matrix<float>(thrust::raw_pointer_cast(data_.data()),
                                                  dataset_size_, dimension_));
    flann_index_.reset(new flann::Index<flann::L2<float>>(
            *flann_dataset_, flann::KDTreeCuda3dIndexParams()));
    flann_index_->buildIndex();
    return true;
}

template int KDTreeFlann::Search<Eigen::Vector3f>(
        const thrust::device_vector<Eigen::Vector3f> &query,
        const KDTreeSearchParam &param,
        thrust::device_vector<KNNIndices> &indices,
        thrust::device_vector<KNNDistances> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector3f>(
        const thrust::device_vector<Eigen::Vector3f> &query,
        int knn,
        thrust::device_vector<KNNIndices> &indices,
        thrust::device_vector<KNNDistances> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector3f>(
        const thrust::device_vector<Eigen::Vector3f> &query,
        float radius,
        thrust::device_vector<KNNIndices> &indices,
        thrust::device_vector<KNNDistances> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::Vector3f>(
        const thrust::device_vector<Eigen::Vector3f> &query,
        float radius,
        int max_nn,
        thrust::device_vector<KNNIndices> &indices,
        thrust::device_vector<KNNDistances> &distance2) const;

template int KDTreeFlann::Search<Eigen::VectorXf>(
        const thrust::device_vector<Eigen::VectorXf> &query,
        const KDTreeSearchParam &param,
        thrust::device_vector<KNNIndices> &indices,
        thrust::device_vector<KNNDistances> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::VectorXf>(
        const thrust::device_vector<Eigen::VectorXf> &query,
        int knn,
        thrust::device_vector<KNNIndices> &indices,
        thrust::device_vector<KNNDistances> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::VectorXf>(
        const thrust::device_vector<Eigen::VectorXf> &query,
        float radius,
        thrust::device_vector<KNNIndices> &indices,
        thrust::device_vector<KNNDistances> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::VectorXf>(
        const thrust::device_vector<Eigen::VectorXf> &query,
        float radius,
        int max_nn,
        thrust::device_vector<KNNIndices> &indices,
        thrust::device_vector<KNNDistances> &distance2) const;
