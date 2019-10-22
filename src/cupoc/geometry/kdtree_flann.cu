
#include "cupoc/geometry/kdtree_flann.h"
#include <flann/flann.hpp>
#include "cupoc/utility/console.h"


namespace cupoc {
namespace geometry {

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
int KDTreeFlann::Search(const T &query,
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
int KDTreeFlann::SearchKNN(const T &query,
                           int knn,
                           thrust::device_vector<int> &indices,
                           thrust::device_vector<float> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Other flann::Index::knnSearch() implementations lose performance due to
    // memory allocation/deallocation.
    if (data_.empty() || dataset_size_ <= 0 ||
        size_t(query.rows()) != dimension_ || knn < 0) {
        return -1;
    }
    flann::Matrix<float> query_flann((float *)query.data(), 1, dimension_);
    indices.resize(knn);
    distance2.resize(knn);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices.data()), query_flann.rows, knn);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2.data()), query_flann.rows, knn);
    int k = flann_index_->knnSearch(query_flann, indices_flann, dists_flann,
                                    knn, flann::SearchParams(-1, 0.0));
    indices.resize(k);
    distance2.resize(k);
    return k;
}

template <typename T>
int KDTreeFlann::SearchRadius(const T &query,
                              float radius,
                              thrust::device_vector<int> &indices,
                              thrust::device_vector<float> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Since max_nn is not given, we let flann to do its own memory management.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory management and CPU caching.
    if (data_.empty() || dataset_size_ <= 0 ||
        size_t(query.rows()) != dimension_) {
        return -1;
    }
    flann::Matrix<float> query_flann((float *)query.data(), 1, dimension_);
    flann::SearchParams param(-1, 0.0);
    param.max_neighbors = -1;
    std::vector<std::vector<int>> indices_vec(1);
    std::vector<std::vector<float>> dists_vec(1);
    int k = flann_index_->radiusSearch(query_flann, indices_vec, dists_vec,
                                       float(radius * radius), param);
    indices = indices_vec[0];
    distance2 = dists_vec[0];
    return k;
}

template <typename T>
int KDTreeFlann::SearchHybrid(const T &query,
                              float radius,
                              int max_nn,
                              thrust::device_vector<int> &indices,
                              thrust::device_vector<float> &distance2) const {
    // This is optimized code for heavily repeated search.
    // It is also the recommended setting for search.
    // Other flann::Index::radiusSearch() implementations lose performance due
    // to memory allocation/deallocation.
    if (data_.empty() || dataset_size_ <= 0 ||
        size_t(query.rows()) != dimension_ || max_nn < 0) {
        return -1;
    }
    flann::Matrix<float> query_flann((float *)query.data(), 1, dimension_);
    flann::SearchParams param(-1, 0.0);
    param.max_neighbors = max_nn;
    indices.resize(max_nn);
    distance2.resize(max_nn);
    flann::Matrix<int> indices_flann(thrust::raw_pointer_cast(indices.data()), query_flann.rows, max_nn);
    flann::Matrix<float> dists_flann(thrust::raw_pointer_cast(distance2.data()), query_flann.rows,
                                      max_nn);
    int k = flann_index_->radiusSearch(query_flann, indices_flann, dists_flann,
                                       float(radius * radius), param);
    indices.resize(k);
    distance2.resize(k);
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
            *flann_dataset_, flann::KDTreeSingleIndexParams(15)));
    flann_index_->buildIndex();
    return true;
}

template int KDTreeFlann::Search<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        const KDTreeSearchParam &param,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        int knn,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        float radius,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        float radius,
        int max_nn,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;

template int KDTreeFlann::Search<Eigen::VectorXf>(
        const Eigen::VectorXf &query,
        const KDTreeSearchParam &param,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchKNN<Eigen::VectorXf>(
        const Eigen::VectorXf &query,
        int knn,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchRadius<Eigen::VectorXf>(
        const Eigen::VectorXf &query,
        float radius,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;
template int KDTreeFlann::SearchHybrid<Eigen::VectorXf>(
        const Eigen::VectorXf &query,
        float radius,
        int max_nn,
        thrust::device_vector<int> &indices,
        thrust::device_vector<float> &distance2) const;

}  // namespace geometry
}  // namespace cupoc