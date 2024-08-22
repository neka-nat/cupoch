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
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <thrust/sort.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/random.h>

#include <Eigen/Geometry>

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace geometry {

namespace {

struct random_functor {
    random_functor(int seed, int n) : seed_(seed), n_(n){};
    const int seed_;
    const int n_;
    __device__ int operator()(size_t idx) const {
        thrust::default_random_engine eng(seed_);
        thrust::uniform_int_distribution<int> dist(0, n_ - 1);
        eng.discard(idx);
        return dist(eng);
    };
};

struct compute_distance_functor {
    compute_distance_functor(const Eigen::Vector4f &plane_model)
        : plane_model_(plane_model){};
    const Eigen::Vector4f plane_model_;
    __device__ float operator()(const Eigen::Vector3f &pt) {
        Eigen::Vector4f point(pt[0], pt[1], pt[2], 1.0);
        return abs(plane_model_.dot(point));
    }
};

Eigen::Vector4f ComputeTrianglePlane(const Eigen::Vector3f &p0,
                                     const Eigen::Vector3f &p1,
                                     const Eigen::Vector3f &p2) {
    const Eigen::Vector3f e0 = p1 - p0;
    const Eigen::Vector3f e1 = p2 - p0;
    Eigen::Vector3f abc = e0.cross(e1);
    float norm = abc.norm();
    // if the three points are co-linear, return invalid plane
    if (norm == 0) {
        return Eigen::Vector4f(0, 0, 0, 0);
    }
    abc /= abc.norm();
    float d = -abc.dot(p0);
    return Eigen::Vector4f(abc(0), abc(1), abc(2), d);
}

}  // namespace

/// \class RANSACResult
///
/// \brief Stores the current best result in the RANSAC algorithm.
class RANSACResult {
public:
    RANSACResult() : fitness_(0), inlier_rmse_(0) {}
    ~RANSACResult() {}

public:
    float fitness_;
    float inlier_rmse_;
};

// Calculates the number of inliers given a list of points and a plane model,
// and the total distance between the inliers and the plane. These numbers are
// then used to evaluate how well the plane model fits the given points.
RANSACResult EvaluateRANSACBasedOnDistance(
        const utility::device_vector<Eigen::Vector3f> &points,
        const Eigen::Vector4f plane_model,
        utility::device_vector<size_t> &inliers,
        float distance_threshold,
        float error) {
    RANSACResult result;
    utility::device_vector<float> errors(points.size());
    inliers.resize(points.size());
    compute_distance_functor func(plane_model);
    auto begin = make_tuple_begin(inliers, errors);
    auto end = thrust::copy_if(
            enumerate_iterator(
                    0, thrust::make_transform_iterator(points.begin(), func)),
            enumerate_iterator(points.size(), thrust::make_transform_iterator(
                                                      points.end(), func)),
            begin,
            [distance_threshold] __device__(
                    const thrust::tuple<size_t, float> &x) {
                return thrust::get<1>(x) < distance_threshold;
            });
    resize_all(thrust::distance(begin, end), inliers, errors);
    error = thrust::reduce(utility::exec_policy(0), errors.begin(),
                           errors.end(), 0.0);

    size_t inlier_num = inliers.size();
    if (inlier_num == 0) {
        result.fitness_ = 0;
        result.inlier_rmse_ = 0;
    } else {
        result.fitness_ = (float)inlier_num / (float)points.size();
        result.inlier_rmse_ = error / std::sqrt((float)inlier_num);
    }
    return result;
}

// Find the plane such that the summed squared distance from the
// plane to all points is minimized.
//
// Reference:
// https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
Eigen::Vector4f GetPlaneFromPoints(
        const utility::device_vector<Eigen::Vector3f> &points,
        const utility::device_vector<size_t> &inliers) {
    Eigen::Vector3f centroid = thrust::reduce(
            utility::exec_policy(0),
            thrust::make_permutation_iterator(points.begin(), inliers.begin()),
            thrust::make_permutation_iterator(points.begin(), inliers.end()),
            Eigen::Vector3f(0.0, 0.0, 0.0));
    centroid /= float(inliers.size());

    Eigen::Vector6f mul_xyz = Eigen::Vector6f::Zero();
    mul_xyz = thrust::transform_reduce(
            utility::exec_policy(0),
            thrust::make_permutation_iterator(points.begin(), inliers.begin()),
            thrust::make_permutation_iterator(points.begin(), inliers.end()),
            [centroid] __device__(const Eigen::Vector3f &pt) -> Eigen::Vector6f {
                Eigen::Vector3f r = pt - centroid;
                Eigen::Vector6f ans;
                ans << r(0) * r(0), r(0) * r(1), r(0) * r(2), r(1) * r(1),
                        r(1) * r(2), r(2) * r(2);
                return ans;
            },
            mul_xyz, thrust::plus<Eigen::Vector6f>());
    float det_x = mul_xyz[3] * mul_xyz[5] - mul_xyz[4] * mul_xyz[4];
    float det_y = mul_xyz[0] * mul_xyz[5] - mul_xyz[2] * mul_xyz[2];
    float det_z = mul_xyz[0] * mul_xyz[3] - mul_xyz[1] * mul_xyz[1];

    Eigen::Vector3f abc;
    if (det_x > det_y && det_x > det_z) {
        abc = Eigen::Vector3f(
                det_x, mul_xyz[2] * mul_xyz[4] - mul_xyz[1] * mul_xyz[5],
                mul_xyz[1] * mul_xyz[4] - mul_xyz[2] * mul_xyz[3]);
    } else if (det_y > det_z) {
        abc = Eigen::Vector3f(
                mul_xyz[2] * mul_xyz[4] - mul_xyz[1] * mul_xyz[5], det_y,
                mul_xyz[1] * mul_xyz[2] - mul_xyz[4] * mul_xyz[0]);
    } else {
        abc = Eigen::Vector3f(mul_xyz[1] * mul_xyz[4] - mul_xyz[2] * mul_xyz[3],
                              mul_xyz[1] * mul_xyz[2] - mul_xyz[4] * mul_xyz[0],
                              det_z);
    }

    float norm = abc.norm();
    // Return invalid plane if the points don't span a plane.
    if (norm == 0) {
        return Eigen::Vector4f::Zero();
    }
    abc /= abc.norm();
    float d = -abc.dot(centroid);
    return Eigen::Vector4f(abc(0), abc(1), abc(2), d);
}

std::tuple<Eigen::Vector4f, utility::device_vector<size_t>>
PointCloud::SegmentPlane(float distance_threshold /* = 0.01 */,
                         size_t ransac_n /* = 3 */,
                         size_t num_iterations /* = 100 */) const {
    RANSACResult result;
    float error = 0.0;

    // Initialize the plane model ax + by + cz + d = 0.
    Eigen::Vector4f plane_model = Eigen::Vector4f(0, 0, 0, 0);
    // Initialize the best plane model.
    Eigen::Vector4f best_plane_model = Eigen::Vector4f(0, 0, 0, 0);

    // Initialize consensus set.
    utility::device_vector<size_t> inliers;
    size_t num_points = points_.size();

    // Return if ransac_n is less than the required plane model parameters.
    if (ransac_n < 3) {
        utility::LogError(
                "ransac_n should be set to higher than or equal to 3.");
        return std::make_tuple(best_plane_model, inliers);
    }
    if (num_points < size_t(ransac_n)) {
        utility::LogError("There must be at least 'ransac_n' points.");
        return std::make_tuple(best_plane_model, inliers);
    }

    utility::device_vector<int> d_cards(num_points);
    utility::device_vector<int> d_keys(num_points);
    thrust::sequence(d_cards.begin(), d_cards.end());
    thrust::host_vector<Eigen::Vector3f> h_pt(3);
    for (int itr = 0; itr < num_iterations; itr++) {
        thrust::tabulate(d_keys.begin(), d_keys.end(),
                         random_functor(rand(), num_points));
        thrust::sort_by_key(utility::exec_policy(0), d_keys.begin(),
                            d_keys.end(), d_cards.begin());
        // Fit model to num_model_parameters randomly selected points among the
        // inliers.
        thrust::copy(thrust::make_permutation_iterator(points_.begin(),
                                                       d_cards.begin()),
                     thrust::make_permutation_iterator(points_.begin(),
                                                       d_cards.begin() + 3),
                     h_pt.begin());
        plane_model = ComputeTrianglePlane(h_pt[0], h_pt[1], h_pt[2]);
        if (plane_model.isZero(0)) {
            continue;
        }

        auto this_result = EvaluateRANSACBasedOnDistance(
                points_, plane_model, inliers, distance_threshold, error);
        if (this_result.fitness_ > result.fitness_ ||
            (this_result.fitness_ == result.fitness_ &&
             this_result.inlier_rmse_ < result.inlier_rmse_)) {
            result = this_result;
            best_plane_model = plane_model;
        }
    }

    // Find the final inliers using best_plane_model.
    inliers.resize(points_.size());
    compute_distance_functor func(best_plane_model);
    auto begin = make_tuple_iterator(inliers.begin(),
                                     thrust::make_discard_iterator());
    auto end = thrust::copy_if(
            enumerate_iterator(
                    0, thrust::make_transform_iterator(points_.begin(), func)),
            enumerate_iterator(points_.size(), thrust::make_transform_iterator(
                                                       points_.end(), func)),
            begin,
            [distance_threshold] __device__(
                    const thrust::tuple<size_t, float> &x) {
                return thrust::get<1>(x) < distance_threshold;
            });
    resize_all(thrust::distance(begin, end), inliers);

    // Improve best_plane_model using the final inliers.
    best_plane_model = GetPlaneFromPoints(points_, inliers);

    utility::LogDebug("RANSAC | Inliers: {:d}, Fitness: {:e}, RMSE: {:e}",
                      inliers.size(), result.fitness_, result.inlier_rmse_);
    return std::make_tuple(best_plane_model, inliers);
}

}  // namespace geometry
}  // namespace cupoch