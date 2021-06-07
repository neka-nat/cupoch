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
#include <Eigen/Geometry>

#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/registration/feature.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace registration {

namespace {

__device__ Eigen::Vector4f ComputePairFeatures(const Eigen::Vector3f &p1,
                                               const Eigen::Vector3f &n1,
                                               const Eigen::Vector3f &p2,
                                               const Eigen::Vector3f &n2) {
    Eigen::Vector4f result;
    Eigen::Vector3f dp2p1 = p2 - p1;
    result(3) = dp2p1.norm();
    if (result(3) == 0.0) {
        return Eigen::Vector4f::Zero();
    }

    auto n1_copy = n1;
    auto n2_copy = n2;
    float angle1 = n1_copy.dot(dp2p1) / result(3);
    float angle2 = n2_copy.dot(dp2p1) / result(3);
    if (acos(fabs(angle1)) > acos(fabs(angle2))) {
        n1_copy = n2;
        n2_copy = n1;
        dp2p1 *= -1.0;
        result(2) = -angle2;
    } else {
        result(2) = angle1;
    }
    auto v = dp2p1.cross(n1_copy);
    float v_norm = v.norm();
    if (v_norm == 0.0) {
        return Eigen::Vector4f::Zero();
    }
    v /= v_norm;
    auto w = n1_copy.cross(v);
    result(1) = v.dot(n2_copy);
    result(0) = atan2(w.dot(n2_copy), n1_copy.dot(n2_copy));
    return result;
}

struct compute_spfh_functor {
    compute_spfh_functor(const Eigen::Vector3f *points,
                         const Eigen::Vector3f *normals,
                         const int *indices,
                         int knn)
        : points_(points), normals_(normals), indices_(indices), knn_(knn){};
    const Eigen::Vector3f *points_;
    const Eigen::Vector3f *normals_;
    const int *indices_;
    const int knn_;
    __device__ Feature<33>::FeatureType operator()(size_t idx) const {
        Feature<33>::FeatureType ft = Feature<33>::FeatureType::Zero();
        int cnt = 0;
        for (size_t k = 0; k < knn_; k++) {
            if (indices_[idx * knn_ + k] >= 0) cnt++;
        }
        float hist_incr = 100.0 / (float)(cnt - 1);
        for (size_t k = 0; k < knn_; k++) {
            const int idx_knn = __ldg(&indices_[idx * knn_ + k]);
            if (idx_knn < 0 || idx == idx_knn) continue;
            // skip the point itself, compute histogram
            auto pf = ComputePairFeatures(points_[idx], normals_[idx],
                                          points_[idx_knn], normals_[idx_knn]);
            int h_index = (int)(floor(11 * (pf(0) + M_PI) / (2.0 * M_PI)));
            if (h_index < 0) h_index = 0;
            if (h_index >= 11) h_index = 10;
            ft[h_index] += hist_incr;
            h_index = (int)(floor(11 * (pf(1) + 1.0) * 0.5));
            if (h_index < 0) h_index = 0;
            if (h_index >= 11) h_index = 10;
            ft[h_index + 11] += hist_incr;
            h_index = (int)(floor(11 * (pf(2) + 1.0) * 0.5));
            if (h_index < 0) h_index = 0;
            if (h_index >= 11) h_index = 10;
            ft[h_index + 22] += hist_incr;
        }
        return ft;
    }
};

std::shared_ptr<Feature<33>> ComputeSPFHFeature(
        const geometry::PointCloud &input,
        const geometry::KDTreeFlann &kdtree,
        const geometry::KDTreeSearchParam &search_param) {
    auto feature = std::make_shared<Feature<33>>();
    feature->Resize((int)input.points_.size());

    utility::device_vector<int> indices;
    utility::device_vector<float> distance2;
    int knn;
    switch (search_param.GetSearchType()) {
        case geometry::KDTreeSearchParam::SearchType::Knn:
            knn = ((const geometry::KDTreeSearchParamKNN &)search_param).knn_;
            break;
        case geometry::KDTreeSearchParam::SearchType::Radius:
            knn = ((const geometry::KDTreeSearchParamRadius &)search_param)
                          .max_nn_;
            break;
        default:
            utility::LogError("Unsupport search param type.");
            return feature;
    }
    kdtree.Search(input.points_, search_param, indices, distance2);
    compute_spfh_functor func(thrust::raw_pointer_cast(input.points_.data()),
                              thrust::raw_pointer_cast(input.normals_.data()),
                              thrust::raw_pointer_cast(indices.data()), knn);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(input.points_.size()),
                      feature->data_.begin(), func);
    return feature;
}

struct compute_fpfh_functor {
    compute_fpfh_functor(const Feature<33>::FeatureType *spfh_data,
                         const int *indices,
                         const float *distance2,
                         int knn)
        : spfh_data_(spfh_data),
          indices_(indices),
          distance2_(distance2),
          knn_(knn){};
    const Feature<33>::FeatureType *spfh_data_;
    const int *indices_;
    const float *distance2_;
    const int knn_;
    __device__ Feature<33>::FeatureType operator()(size_t idx) const {
        Feature<33>::FeatureType ft = Feature<33>::FeatureType::Zero();
        float sum[3] = {0.0, 0.0, 0.0};
        for (size_t k = 0; k < knn_; k++) {
            // skip the point itself
            int idx_knn = indices_[idx * knn_ + k];
            if (idx_knn < 0 || idx == idx_knn) continue;
            float dist = distance2_[idx * knn_ + k];
            if (dist == 0.0) continue;
#pragma unroll
            for (int j = 0; j < 33; j++) {
                float val = spfh_data_[idx_knn][j] / dist;
                sum[j / 11] += val;
                ft[j] += val;
            }
        }
#pragma unroll
        for (int j = 0; j < 3; j++)
            if (sum[j] != 0.0) sum[j] = 100.0 / sum[j];
#pragma unroll
        for (int j = 0; j < 33; j++) {
            ft[j] *= sum[j / 11];
            // The commented line is the fpfh function in the paper.
            // But according to PCL implementation, it is skipped.
            // Our initial test shows that the full fpfh function in the
            // paper seems to be better than PCL implementation. Further
            // test required.
            ft[j] += spfh_data_[idx][j];
        }
        return ft;
    }
};

}  // namespace

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

std::shared_ptr<Feature<33>> ComputeFPFHFeature(
        const geometry::PointCloud &input,
        const geometry::KDTreeSearchParam
                &search_param /* = geometry::KDTreeSearchParamKNN()*/) {
    auto feature = std::make_shared<Feature<33>>();
    feature->Resize((int)input.points_.size());

    if (!input.HasNormals()) {
        utility::LogError(
                "[ComputeFPFHFeature] Failed because input point cloud has no "
                "normal.");
    }

    geometry::KDTreeFlann kdtree(input);
    auto spfh = ComputeSPFHFeature(input, kdtree, search_param);
    utility::device_vector<int> indices;
    utility::device_vector<float> distance2;
    int knn;
    switch (search_param.GetSearchType()) {
        case geometry::KDTreeSearchParam::SearchType::Knn:
            knn = ((const geometry::KDTreeSearchParamKNN &)search_param).knn_;
            break;
        case geometry::KDTreeSearchParam::SearchType::Radius:
            knn = ((const geometry::KDTreeSearchParamRadius &)search_param)
                          .max_nn_;
            break;
        default:
            utility::LogError("Unsupport search param type.");
            return feature;
    }
    kdtree.Search(input.points_, search_param, indices, distance2);
    compute_fpfh_functor func(thrust::raw_pointer_cast(spfh->data_.data()),
                              thrust::raw_pointer_cast(indices.data()),
                              thrust::raw_pointer_cast(distance2.data()), knn);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(input.points_.size()),
                      feature->data_.begin(), func);
    return feature;
}

}  // namespace registration
}  // namespace cupoch