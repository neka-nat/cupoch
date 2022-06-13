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
#include <thrust/iterator/discard_iterator.h>
#include <Eigen/Geometry>

#include "cupoch/geometry/geometry_functor.h"
#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/geometry/keypoint.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/eigenvalue.h"
#include "cupoch/utility/range.h"

namespace cupoch {

namespace {

float ComputeModelResolution(const geometry::PointCloud& points,
                             const geometry::KDTreeFlann& kdtree) {
    utility::device_vector<int> indices;
    utility::device_vector<float> distance2;
    kdtree.SearchKNN(points.points_, 2, indices, distance2);
    thrust::strided_range<
            utility::device_vector<float>::const_iterator>
            range_dists(distance2.begin() + 1, distance2.end(), 2);
    float resolution = thrust::reduce(utility::exec_policy(utility::GetStream(0))
            ->on(utility::GetStream(0)),
            range_dists.begin(), range_dists.end(), 0.0f);
    resolution /= points.points_.size();
    return std::sqrt(resolution);
}

__device__ Eigen::Vector3f ComputeThirdEigenValue(const Eigen::Matrix<float, 9, 1> &cum,
                                                  int count) {
    Eigen::Matrix<float, 9, 1> cumulants = cum / (float)count;
    Eigen::Matrix3f covariance;
    covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
    covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
    covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
    covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
    covariance(1, 0) = covariance(0, 1);
    covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
    covariance(2, 0) = covariance(0, 2);
    covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
    covariance(2, 1) = covariance(1, 2);
    if (covariance.isZero()) return Eigen::Vector3f::Constant(-1.0);

    return utility::FastEigen3x3Val(covariance);
}

struct compute_third_eigen_values_functor {
    compute_third_eigen_values_functor(int min_neighbors, float gamma_21, float gamma_32)
    : min_neighbors_(min_neighbors), gamma_21_(gamma_21), gamma_32_(gamma_32) {};
    const int min_neighbors_;
    const float gamma_21_;
    const float gamma_32_;
    __device__ float operator()(
            const thrust::tuple<Eigen::Matrix<float, 9, 1>, int> &x) const {
        if (thrust::get<1>(x) < min_neighbors_) return -1.0;
        Eigen::Vector3f eigs = ComputeThirdEigenValue(thrust::get<0>(x), thrust::get<1>(x));
        return (eigs[2] > 0 && eigs[1] / eigs[2] < gamma_21_ && eigs[0] / eigs[1] < gamma_32_) ? eigs[0] : -1.0;
    }
};

struct is_local_maxima_functor {
    is_local_maxima_functor(const int *indices, const float *third_eigen_values, int knn)
    : indices_(indices), third_eigen_values_(third_eigen_values), knn_(knn) {}
    const int *indices_;
    const float *third_eigen_values_;
    const int knn_;
    __device__ bool operator() (size_t idx) {
        const float query = third_eigen_values_[idx];
        if (query < 0) return false;
        for (int k = 0; k < knn_; ++k) {
            const int l = indices_[idx * knn_ + k];
            if (l < 0) continue;
            if (query < third_eigen_values_[l]) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace

namespace geometry {
namespace keypoint {

std::tuple<std::shared_ptr<PointCloud>, std::shared_ptr<utility::device_vector<bool>>>
ComputeISSKeypoints(
        const PointCloud& input,
        float salient_radius /* = 0.0 */,
        float non_max_radius /* = 0.0 */,
        float gamma_21 /* = 0.975 */,
        float gamma_32 /* = 0.975 */,
        int min_neighbors /*= 5 */,
        int max_neighbors /*= NUM_MAX_NN */) {
    if (input.points_.empty()) {
        utility::LogWarning("[ComputeISSKeypoints] Input PointCloud is empty!");
        return std::make_tuple(std::make_shared<PointCloud>(), std::make_shared<utility::device_vector<bool>>());
    }

    KDTreeFlann kdtree;
    kdtree.SetGeometry(input);
    if (salient_radius == 0.0 || non_max_radius == 0.0) {
        const float resolution = ComputeModelResolution(input, kdtree);
        salient_radius = 6 * resolution;
        non_max_radius = 4 * resolution;
        utility::LogDebug(
                "[ComputeISSKeypoints] Computed salient_radius = {}, "
                "non_max_radius = {} from input model",
                salient_radius, non_max_radius);
    }
    utility::device_vector<int> indices;
    utility::device_vector<float> distance2;
    kdtree.SearchRadius(input.points_, salient_radius, max_neighbors, indices,
                        distance2);

    const size_t n_pt = input.points_.size();
    utility::device_vector<float> third_eigen_values(n_pt, -1);
    utility::device_vector<Eigen::Matrix<float, 9, 1>> cumulants(n_pt);
    utility::device_vector<int> counts(n_pt);
    thrust::repeated_range<thrust::counting_iterator<size_t>> range(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator(n_pt), max_neighbors);
    thrust::reduce_by_key(
            utility::exec_policy(0)->on(0), range.begin(), range.end(),
            thrust::make_transform_iterator(
                    indices.begin(),
                    geometry::compute_cumulant_functor(
                            thrust::raw_pointer_cast(input.points_.data()))),
            thrust::make_discard_iterator(),
            make_tuple_begin(cumulants, counts), thrust::equal_to<size_t>(),
            add_tuple_functor<Eigen::Matrix<float, 9, 1>, int>());
    thrust::transform(make_tuple_begin(cumulants, counts),
                      make_tuple_end(cumulants, counts), third_eigen_values.begin(),
                      compute_third_eigen_values_functor(min_neighbors, gamma_21, gamma_32));

    auto mask = std::make_shared<utility::device_vector<bool>>(n_pt);
    kdtree.SearchRadius(input.points_, non_max_radius, max_neighbors, indices,
                        distance2);
    is_local_maxima_functor func(thrust::raw_pointer_cast(indices.data()),
                                 thrust::raw_pointer_cast(third_eigen_values.data()),
                                 max_neighbors);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_pt),
                      mask->begin(), func);

    auto out = input.SelectByMask(*mask);
    utility::LogDebug("[ComputeISSKeypoints] Extracted {} keypoints",
                      out->points_.size());
    return std::make_tuple(std::move(out), std::move(mask));
}

}  // namespace keypoint
}  // namespace geometry
}  // namespace cupoch