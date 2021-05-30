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
#include "cupoch/utility/range.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

__device__ float signf(float x) { return x / fabs(x); }

__device__ Eigen::Vector3f ComputeEigenvector0(const Eigen::Matrix3f &A,
                                               float eval0) {
    Eigen::Vector3f row0(A(0, 0) - eval0, A(0, 1), A(0, 2));
    Eigen::Vector3f row1(A(0, 1), A(1, 1) - eval0, A(1, 2));
    Eigen::Vector3f row2(A(0, 2), A(1, 2), A(2, 2) - eval0);
    Eigen::Vector3f rxr[3];
    rxr[0] = row0.cross(row1);
    rxr[1] = row0.cross(row2);
    rxr[2] = row1.cross(row2);
    Eigen::Vector3f d;
    d[0] = rxr[0].dot(rxr[0]);
    d[1] = rxr[1].dot(rxr[1]);
    d[2] = rxr[2].dot(rxr[2]);

    int imax;
    d.maxCoeff(&imax);
    return rxr[imax] / sqrtf(d[imax]);
}

__device__ Eigen::Vector3f ComputeEigenvector1(const Eigen::Matrix3f &A,
                                               const Eigen::Vector3f &evec0,
                                               float eval1) {
    float max_evec0_abs = max(fabs(evec0(0)), fabs(evec0(1)));
    float inv_length =
            1 / sqrtf(max_evec0_abs * max_evec0_abs + evec0(2) * evec0(2));
    Eigen::Vector3f U = (fabs(evec0(0)) > fabs(evec0(1)))
                                ? Eigen::Vector3f(-evec0(2), 0, evec0(0))
                                : Eigen::Vector3f(0, evec0(2), -evec0(1));
    U *= inv_length;
    Eigen::Vector3f V = evec0.cross(U);

    Eigen::Vector3f AU(A(0, 0) * U(0) + A(0, 1) * U(1) + A(0, 2) * U(2),
                       A(0, 1) * U(0) + A(1, 1) * U(1) + A(1, 2) * U(2),
                       A(0, 2) * U(0) + A(1, 2) * U(1) + A(2, 2) * U(2));

    Eigen::Vector3f AV = {A(0, 0) * V(0) + A(0, 1) * V(1) + A(0, 2) * V(2),
                          A(0, 1) * V(0) + A(1, 1) * V(1) + A(1, 2) * V(2),
                          A(0, 2) * V(0) + A(1, 2) * V(1) + A(2, 2) * V(2)};

    float m00 = U(0) * AU(0) + U(1) * AU(1) + U(2) * AU(2) - eval1;
    float m01 = U(0) * AV(0) + U(1) * AV(1) + U(2) * AV(2);
    float m11 = V(0) * AV(0) + V(1) * AV(1) + V(2) * AV(2) - eval1;

    float absM00 = fabs(m00);
    float absM01 = fabs(m01);
    float absM11 = fabs(m11);
    float max_abs_comp0 = max(absM00, absM11);
    float max_abs_comp = max(max_abs_comp0, absM01);
    float coef2 = min(max_abs_comp0, absM01) / max(max_abs_comp, 1.0e-6);
    float coef1 = 1.0 / sqrtf(1.0 + coef2 * coef2);
    if (absM00 >= absM11) {
        coef2 *= coef1 * signf(m00) * signf(m01);
        return (max_abs_comp0 >= absM01) ? coef2 * U - coef1 * V
                                         : coef1 * U - coef2 * V;
    } else {
        coef2 *= coef1 * signf(m11) * signf(m01);
        return (max_abs_comp0 >= absM01) ? coef1 * U - coef2 * V
                                         : coef2 * U - coef1 * V;
    }
}

__device__ Eigen::Vector3f FastEigen3x3(Eigen::Matrix3f &A) {
    // Previous version based on:
    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
    // Current version based on
    // https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    // which handles edge cases like points on a plane

    float max_coeff = A.maxCoeff();
    if (max_coeff == 0) {
        return Eigen::Vector3f::Zero();
    }
    A /= max_coeff;

    float norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    if (norm > 0) {
        Eigen::Vector3f eval;
        Eigen::Vector3f evec0;
        Eigen::Vector3f evec1;
        Eigen::Vector3f evec2;

        float q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;

        float b00 = A(0, 0) - q;
        float b11 = A(1, 1) - q;
        float b22 = A(2, 2) - q;

        float p = sqrtf((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

        float c00 = b11 * b22 - A(1, 2) * A(1, 2);
        float c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
        float c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
        float det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);

        float half_det = det * 0.5;
        half_det = min(max(half_det, -1.0), 1.0);

        float angle = acos(half_det) / (float)3;
        float const two_thirds_pi = 2.09439510239319549;
        float beta2 = cos(angle) * 2;
        float beta0 = cos(angle + two_thirds_pi) * 2;
        float beta1 = -(beta0 + beta2);

        eval(0) = q + p * beta0;
        eval(1) = q + p * beta1;
        eval(2) = q + p * beta2;

        if (half_det >= 0) {
            evec2 = ComputeEigenvector0(A, eval(2));
            if (eval(2) < eval(0) && eval(2) < eval(1)) {
                A *= max_coeff;
                return evec2;
            }
            evec1 = ComputeEigenvector1(A, evec2, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                return evec1;
            }
            evec0 = evec1.cross(evec2);
            return evec0;
        } else {
            evec0 = ComputeEigenvector0(A, eval(0));
            if (eval(0) < eval(1) && eval(0) < eval(2)) {
                A *= max_coeff;
                return evec0;
            }
            evec1 = ComputeEigenvector1(A, evec0, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                return evec1;
            }
            evec2 = evec0.cross(evec1);
            return evec2;
        }
    } else {
        A *= max_coeff;
        int min_id;
        A.diagonal().minCoeff(&min_id);
        Eigen::Vector3f unit = Eigen::Vector3f::Zero();
        unit[min_id] = 1.0;
        return unit;
    }
}

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

    return FastEigen3x3(covariance);
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
