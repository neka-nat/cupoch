
#include "cupoch/geometry/kdtree_flann.h"
#define FLANN_USE_CUDA
#include <flann/flann.hpp>
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct convert_float4_functor {
    __device__
    float4 operator() (const Eigen::Vector3f& x) const {
        return make_float4(x[0], x[1], x[2], 0.0f);
    }
};

}


KDTreeFlann::KDTreeFlann() {}

KDTreeFlann::KDTreeFlann(const PointCloud &data) {SetGeometry(data);}

KDTreeFlann::~KDTreeFlann() {}

bool KDTreeFlann::SetGeometry(const PointCloud &geometry) {
    return SetRawData(geometry.points_);
}

template <typename T>
int KDTreeFlann::Search(const thrust::device_vector<T> &query,
                        const KDTreeSearchParam &param,
                        thrust::device_vector<int> &indices,
                        thrust::device_vector<float> &distance2) const {
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
                           thrust::device_vector<int> &indices,
                           thrust::device_vector<float> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Other flann::Index::knnSearch() implementations lose performance due to
    // memory allocation/deallocation.
    if (data_.empty() || query.empty() || dataset_size_ <= 0 || knn < 0 || knn > NUM_MAX_NN) return -1;
    T query0 = query[0];
    if (size_t(query0.size()) != dimension_) return -1;
    convert_float4_functor func;
    thrust::device_vector<float4> query_f4(query.size());
    thrust::transform(query.begin(), query.end(), query_f4.begin(), func);
    flann::Matrix<float> query_flann((float *)(thrust::raw_pointer_cast(query_f4.data())), query.size(), dimension_, sizeof(float) * 4);
    const int total_size = query.size() * knn;
    indices.resize(total_size);
    distance2.resize(total_size);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices.data()), query_flann.rows, knn);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2.data()), query_flann.rows, knn);
    flann::SearchParams param;
    param.matrices_in_gpu_ram = true;
    int k = flann_index_->knnSearch(query_flann, indices_flann, dists_flann,
                                    knn, param);
    return k;
}

template <typename T>
int KDTreeFlann::SearchRadius(const thrust::device_vector<T> &query,
                              float radius,
                              thrust::device_vector<int> &indices,
                              thrust::device_vector<float> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Since max_nn is not given, we let flann to do its own memory management.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory management and CPU caching.
    if (data_.empty() || query.empty() || dataset_size_ <= 0) return -1;
    T query0 = query[0];
    if (size_t(query0.size()) != dimension_) return -1;
    convert_float4_functor func;
    thrust::device_vector<float4> query_f4(query.size());
    thrust::transform(query.begin(), query.end(), query_f4.begin(), func);
    flann::Matrix<float> query_flann((float *)(thrust::raw_pointer_cast(query_f4.data())), query.size(), dimension_, sizeof(float) * 4);
    flann::SearchParams param(-1, 0.0);
    param.max_neighbors = NUM_MAX_NN;
    param.matrices_in_gpu_ram = true;
    indices.resize(query.size() * NUM_MAX_NN);
    distance2.resize(query.size() * NUM_MAX_NN);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices.data()), query_flann.rows, NUM_MAX_NN);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2.data()), query_flann.rows, NUM_MAX_NN);
    int k = flann_index_->radiusSearch(query_flann, indices_flann, dists_flann,
                                       float(radius * radius), param);
    return k;
}

template <typename T>
int KDTreeFlann::SearchHybrid(const thrust::device_vector<T> &query,
                              float radius,
                              int max_nn,
                              thrust::device_vector<int> &indices,
                              thrust::device_vector<float> &distance2) const {
    // This is optimized code for heavily repeated search.
    // It is also the recommended setting for search.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory allocation/deallocation.
    if (data_.empty() || query.empty() || dataset_size_ <= 0 || max_nn < 0 || max_nn > NUM_MAX_NN) return -1;
    T query0 = query[0];
    if (size_t(query0.size()) != dimension_) return -1;
    convert_float4_functor func;
    thrust::device_vector<float4> query_f4(query.size());
    thrust::transform(query.begin(), query.end(), query_f4.begin(), func);
    flann::Matrix<float> query_flann((float *)(thrust::raw_pointer_cast(query_f4.data())), query.size(), dimension_, sizeof(float) * 4);
    flann::SearchParams param(-1, 0.0);
    param.max_neighbors = max_nn;
    param.matrices_in_gpu_ram = true;
    indices.resize(query.size() * max_nn);
    distance2.resize(query.size() * max_nn);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices.data()), query_flann.rows, max_nn);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2.data()), query_flann.rows, max_nn);
    int k = flann_index_->radiusSearch(query_flann, indices_flann, dists_flann,
                                       float(radius * radius), param);
    return k;
}

template <typename T>
bool KDTreeFlann::SetRawData(const thrust::device_vector<T> &data) {
    dimension_ = T::SizeAtCompileTime;
    dataset_size_ = data.size();
    if (dimension_ == 0 || dataset_size_ == 0) {
        utility::LogWarning(
                "[KDTreeFlann::SetRawData] Failed due to no data.\n");
        return false;
    }
    data_.resize(dataset_size_);
    convert_float4_functor func;
    thrust::transform(data.begin(), data.end(), data_.begin(), func);
    flann_dataset_.reset(new flann::Matrix<float>((float*)thrust::raw_pointer_cast(data_.data()),
                                                  dataset_size_, dimension_, sizeof(float) * 4));
    flann::KDTreeCuda3dIndexParams index_params;
    flann_index_.reset(new flann::KDTreeCuda3dIndex<flann::L2<float>>(
            *flann_dataset_, index_params));
    flann_index_->buildIndex();
    return true;
}

template <typename T>
int KDTreeFlann::SearchKNN(const T &query,
                           int knn,
                           thrust::host_vector<int> &indices,
                           thrust::host_vector<float> &distance2) const {
    thrust::device_vector<T> query_dv(1, query);
    thrust::device_vector<int> indices_dv;
    thrust::device_vector<float> distance2_dv;
    auto result = SearchKNN<T>(query_dv, knn, indices_dv, distance2_dv);
    indices = indices_dv;
    distance2 = distance2_dv;
    return result;
}

template <typename T>
int KDTreeFlann::SearchRadius(const T &query,
                              float radius,
                              thrust::host_vector<int> &indices,
                              thrust::host_vector<float> &distance2) const {
    thrust::device_vector<T> query_dv(1, query);
    thrust::device_vector<int> indices_dv;
    thrust::device_vector<float> distance2_dv;
    auto result = SearchRadius<T>(query_dv, radius, indices_dv, distance2_dv);
    indices = indices_dv;
    distance2 = distance2_dv;
    return result;
}

template <typename T>
int KDTreeFlann::SearchHybrid(const T &query,
                              float radius,
                              int max_nn,
                              thrust::host_vector<int> &indices,
                              thrust::host_vector<float> &distance2) const {
    thrust::device_vector<T> query_dv(1, query);
    thrust::device_vector<int> indices_dv;
    thrust::device_vector<float> distance2_dv;
    auto result = SearchHybrid<T>(query_dv, radius, max_nn, indices_dv, distance2_dv);
    indices = indices_dv;
    distance2 = distance2_dv;
    return result;
}

template int KDTreeFlann::Search<Eigen::Vector3f>(
        const thrust::device_vector<Eigen::Vector3f> &query,
        const KDTreeSearchParam &param,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector3f>(
        const thrust::device_vector<Eigen::Vector3f> &query,
        int knn,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector3f>(
        const thrust::device_vector<Eigen::Vector3f> &query,
        float radius,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::Vector3f>(
        const thrust::device_vector<Eigen::Vector3f> &query,
        float radius,
        int max_nn,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        int knn,
        thrust::host_vector<int> &indices,
        thrust::host_vector<float> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        float radius,
        thrust::host_vector<int> &indices,
        thrust::host_vector<float> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        float radius,
        int max_nn,
        thrust::host_vector<int> &indices,
        thrust::host_vector<float> &distance2) const;
template bool KDTreeFlann::SetRawData<Eigen::Vector3f>(
        const thrust::device_vector<Eigen::Vector3f> &data);
        