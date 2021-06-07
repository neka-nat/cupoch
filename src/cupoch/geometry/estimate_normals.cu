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
#include <thrust/iterator/discard_iterator.h>

#include <Eigen/Geometry>

#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/eigenvalue.h"
#include "cupoch/utility/range.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

__device__ Eigen::Vector3f ComputeNormal(const Eigen::Matrix<float, 9, 1> &cum,
                                         int count) {
    if (count < 3) return Eigen::Vector3f(0.0, 0.0, 1.0);
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

    return utility::FastEigen3x3(covariance);
}

struct compute_normal_functor {
    compute_normal_functor(){};
    __device__ Eigen::Vector3f operator()(
            const thrust::tuple<Eigen::Matrix<float, 9, 1>, int> &x) const {
        Eigen::Vector3f normal =
                ComputeNormal(thrust::get<0>(x), thrust::get<1>(x));
        return (normal.norm() == 0.0) ? Eigen::Vector3f(0.0, 0.0, 1.0) : normal;
    }
};

struct compute_cumulant_functor {
    compute_cumulant_functor(const Eigen::Vector3f *points) : points_(points){};
    const Eigen::Vector3f *points_;
    __device__ thrust::tuple<Eigen::Matrix<float, 9, 1>, int> operator()(
            int idx) const {
        Eigen::Matrix<float, 9, 1> cm;
        cm.setZero();
        if (idx < 0) return thrust::make_tuple(cm, 0);
        const Eigen::Vector3f point = points_[idx];
        cm(0) = point(0);
        cm(1) = point(1);
        cm(2) = point(2);
        cm(3) = point(0) * point(0);
        cm(4) = point(0) * point(1);
        cm(5) = point(0) * point(2);
        cm(6) = point(1) * point(1);
        cm(7) = point(1) * point(2);
        cm(8) = point(2) * point(2);
        return thrust::make_tuple(cm, 1);
    }
};

struct align_normals_direction_functor {
    align_normals_direction_functor(
            const Eigen::Vector3f &orientation_reference)
        : orientation_reference_(orientation_reference){};
    const Eigen::Vector3f orientation_reference_;
    __device__ void operator()(Eigen::Vector3f &normal) const {
        if (normal.norm() == 0.0) {
            normal = orientation_reference_;
        } else if (normal.dot(orientation_reference_) < 0.0) {
            normal *= -1.0;
        }
    }
};

}  // namespace

bool PointCloud::EstimateNormals(const KDTreeSearchParam &search_param) {
    if (HasNormals() == false) {
        normals_.resize(points_.size());
    }
    KDTreeFlann kdtree;
    kdtree.SetGeometry(*this);
    utility::device_vector<int> indices;
    utility::device_vector<float> distance2;
    kdtree.Search(points_, search_param, indices, distance2);
    int knn;
    switch (search_param.GetSearchType()) {
        case KDTreeSearchParam::SearchType::Knn:
            knn = ((const KDTreeSearchParamKNN &)search_param).knn_;
            break;
        case KDTreeSearchParam::SearchType::Radius:
            knn = ((const KDTreeSearchParamRadius &)search_param).max_nn_;
            break;
        default:
            utility::LogError("Unknown search param type.");
            return false;
    }
    if (knn <= 0) {
        thrust::fill(normals_.begin(), normals_.end(),
                     Eigen::Vector3f(0.0, 0.0, 1.0));
        return true;
    }
    size_t n_pt = points_.size();
    utility::device_vector<Eigen::Matrix<float, 9, 1>> cumulants(n_pt);
    utility::device_vector<int> counts(n_pt);
    thrust::repeated_range<thrust::counting_iterator<size_t>> range(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator(n_pt), knn);
    thrust::reduce_by_key(
            utility::exec_policy(0)->on(0), range.begin(), range.end(),
            thrust::make_transform_iterator(
                    indices.begin(),
                    compute_cumulant_functor(
                            thrust::raw_pointer_cast(points_.data()))),
            thrust::make_discard_iterator(),
            make_tuple_begin(cumulants, counts), thrust::equal_to<size_t>(),
            add_tuple_functor<Eigen::Matrix<float, 9, 1>, int>());
    thrust::transform(make_tuple_begin(cumulants, counts),
                      make_tuple_end(cumulants, counts), normals_.begin(),
                      compute_normal_functor());
    return true;
}

bool PointCloud::OrientNormalsToAlignWithDirection(
        const Eigen::Vector3f &orientation_reference) {
    if (HasNormals() == false) {
        utility::LogWarning(
                "[OrientNormalsToAlignWithDirection] No normals in the "
                "PointCloud. Call EstimateNormals() first.\n");
        return false;
    }
    align_normals_direction_functor func(orientation_reference);
    thrust::for_each(normals_.begin(), normals_.end(), func);
    return true;
}
