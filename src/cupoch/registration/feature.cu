#include <Eigen/Geometry>
#include "cupoch/registration/feature.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::registration;

namespace {

__device__
Eigen::Vector4f ComputePairFeatures(const Eigen::Vector3f &p1,
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
    compute_spfh_functor(const Eigen::Vector3f* points,
                         const Eigen::Vector3f* normals,
                         const int* indices, int knn, float hist_incr)
        : points_(points), normals_(normals), indices_(indices), knn_(knn), hist_incr_(hist_incr) {};
    const Eigen::Vector3f* points_;
    const Eigen::Vector3f* normals_;
    const int* indices_;
    const int knn_;
    const float hist_incr_;
    __device__
    Feature<33>::FeatureType operator()(size_t idx) const {
        Feature<33>::FeatureType ft;
        for (size_t k = 1; k < knn_; k++) {
            // skip the point itself, compute histogram
            auto pf = ComputePairFeatures(points_[idx], normals_[idx],
                                          points_[indices_[idx * knn_ + k]],
                                          normals_[indices_[idx * knn_ + k]]);
            int h_index = (int)(floor(11 * (pf(0) + M_PI) / (2.0 * M_PI)));
            if (h_index < 0) h_index = 0;
            if (h_index >= 11) h_index = 10;
            ft[h_index] += hist_incr_;
            h_index = (int)(floor(11 * (pf(1) + 1.0) * 0.5));
            if (h_index < 0) h_index = 0;
            if (h_index >= 11) h_index = 10;
            ft[h_index + 11] += hist_incr_;
            h_index = (int)(floor(11 * (pf(2) + 1.0) * 0.5));
            if (h_index < 0) h_index = 0;
            if (h_index >= 11) h_index = 10;
            ft[h_index + 22] += hist_incr_;
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
    auto knn = ((const geometry::KDTreeSearchParamKNN &)search_param).knn_;
    kdtree.SearchKNN(input.points_, knn,
                     indices, distance2);
    float hist_incr = 100.0 / (float)(knn - 1);
    compute_spfh_functor func(thrust::raw_pointer_cast(input.points_.data()),
                              thrust::raw_pointer_cast(input.normals_.data()),
                              thrust::raw_pointer_cast(indices.data()),
                              knn, hist_incr);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(input.points_.size()),
                      feature->data_.begin(), func);
    return feature;
}

struct compute_fpfh_functor {
    compute_fpfh_functor(const Feature<33>::FeatureType* spfh_data,
                         const int* indices, const float* distance2, int knn)
        : spfh_data_(spfh_data), indices_(indices), distance2_(distance2), knn_(knn) {};
    const Feature<33>::FeatureType* spfh_data_;
    const int* indices_;
    const float* distance2_;
    const int knn_;
    __device__
    Feature<33>::FeatureType operator() (size_t idx) const {
        Feature<33>::FeatureType ft;
        float sum[3] = {0.0, 0.0, 0.0};
        for (size_t k = 1; k < knn_; k++) {
            // skip the point itself
            float dist = distance2_[idx * knn_ + k];
            if (dist == 0.0) continue;
            for (int j = 0; j < 33; j++) {
                float val = spfh_data_[indices_[idx * knn_ + k]][j] / dist;
                sum[j / 11] += val;
                ft[j] += val;
            }
        }
        for (int j = 0; j < 3; j++)
            if (sum[j] != 0.0) sum[j] = 100.0 / sum[j];
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

}

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
    kdtree.SearchKNN(input.points_,
                     ((const geometry::KDTreeSearchParamKNN &)search_param).knn_,
                     indices, distance2);
    compute_fpfh_functor func(thrust::raw_pointer_cast(spfh->data_.data()),
                              thrust::raw_pointer_cast(indices.data()),
                              thrust::raw_pointer_cast(distance2.data()),
                              ((const geometry::KDTreeSearchParamKNN &)search_param).knn_);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(input.points_.size()),
                      feature->data_.begin(), func);
    return feature;
}