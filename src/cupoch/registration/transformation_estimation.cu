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
#include <thrust/inner_product.h>

#include <Eigen/Geometry>

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/registration/kabsch.h"
#include "cupoch/registration/transformation_estimation.h"

using namespace cupoch;
using namespace cupoch::registration;

namespace {

struct pt2pl_jacobian_residual_functor
    : public utility::jacobian_residual_functor<Eigen::Vector6f> {
    pt2pl_jacobian_residual_functor(const Eigen::Vector3f *source,
                                    const Eigen::Vector3f *target_points,
                                    const Eigen::Vector3f *target_normals,
                                    const Eigen::Vector2i *corres)
        : source_(source),
          target_points_(target_points),
          target_normals_(target_normals),
          corres_(corres){};
    const Eigen::Vector3f *source_;
    const Eigen::Vector3f *target_points_;
    const Eigen::Vector3f *target_normals_;
    const Eigen::Vector2i *corres_;
    __device__ void operator()(int idx, Eigen::Vector6f &vec, float &r) const {
        const Eigen::Vector3f &vs = source_[corres_[idx][0]];
        const Eigen::Vector3f &vt = target_points_[corres_[idx][1]];
        const Eigen::Vector3f &nt = target_normals_[corres_[idx][1]];
        r = (vs - vt).dot(nt);
        vec.block<3, 1>(0, 0) = vs.cross(nt);
        vec.block<3, 1>(3, 0) = nt;
    }
};

struct symmetric_jacobian_residual_functor
    : public utility::jacobian_residual_functor<Eigen::Vector6f> {
    symmetric_jacobian_residual_functor(const Eigen::Vector3f *source,
                                    const Eigen::Vector3f *source_normals,
                                    const Eigen::Vector3f *target_points,
                                    const Eigen::Vector3f *target_normals,
                                    const Eigen::Vector2i *corres)
        : source_(source),
          source_normals_(source_normals),
          target_points_(target_points),
          target_normals_(target_normals),
          corres_(corres){};
    const Eigen::Vector3f *source_;
    const Eigen::Vector3f *source_normals_;
    const Eigen::Vector3f *target_points_;
    const Eigen::Vector3f *target_normals_;
    const Eigen::Vector2i *corres_;
    __device__ void operator()(int idx, Eigen::Vector6f &vec, float &r) const {
        const Eigen::Vector3f &vs = source_[corres_[idx][0]];
        const Eigen::Vector3f &ns = source_normals_[corres_[idx][0]];
        const Eigen::Vector3f &vt = target_points_[corres_[idx][1]];
        const Eigen::Vector3f &nt = target_normals_[corres_[idx][1]];

        Eigen::Vector3f n = ns + nt;  // sum of normals
        Eigen::Vector3f d = vs - vt;  // difference of points

        r = d.dot(n);  // symmetric residual

        Eigen::Vector3f cross_product = (vs + vt).cross(n);
        vec.block<3, 1>(0, 0) = cross_product;
        vec.block<3, 1>(3, 0) = n;
    }
};

// Define a function or a lambda to compute the error using normals
__device__ float ComputeErrorUsingNormals(
        const Eigen::Vector3f &source_point,
        const Eigen::Vector3f &source_normal,
        const Eigen::Vector3f &target_point,
        const Eigen::Vector3f &target_normal) {
    // Symmetric treatment of normals
    Eigen::Vector3f combined_normal = source_normal + target_normal;

    // Compute the symmetric point-to-plane error
    float error = (source_point - target_point).dot(combined_normal);

    return error * error;  // Return squared error for consistency
}

}  // namespace

float TransformationEstimationPointToPoint::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    const float err = thrust::inner_product(
            thrust::make_permutation_iterator(
                    source.points_.begin(),
                    thrust::make_transform_iterator(
                            corres.begin(),
                            element_get_functor<Eigen::Vector2i, 0>())),
            thrust::make_permutation_iterator(
                    source.points_.begin(),
                    thrust::make_transform_iterator(
                            corres.end(),
                            element_get_functor<Eigen::Vector2i, 0>())),
            thrust::make_permutation_iterator(
                    target.points_.begin(),
                    thrust::make_transform_iterator(
                            corres.begin(),
                            element_get_functor<Eigen::Vector2i, 1>())),
            0.0f, thrust::plus<float>(),
            [] __device__(const Eigen::Vector3f &lhs,
                          const Eigen::Vector3f &rhs) {
                return (lhs - rhs).squaredNorm();
            });
    return std::sqrt(err / (float)corres.size());
}

Eigen::Matrix4f TransformationEstimationPointToPoint::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    return Kabsch(source.points_, target.points_, corres);
}

float TransformationEstimationPointToPlane::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || !target.HasNormals()) return 0.0;
    const float err = thrust::transform_reduce(
            utility::exec_policy(0),
            make_tuple_iterator(
                    thrust::make_permutation_iterator(
                            source.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.begin(),
                                    element_get_functor<Eigen::Vector2i, 0>())),
                    thrust::make_permutation_iterator(
                            target.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.begin(),
                                    element_get_functor<Eigen::Vector2i, 1>())),
                    thrust::make_permutation_iterator(
                            target.normals_.begin(),
                            thrust::make_transform_iterator(
                                    corres.begin(),
                                    element_get_functor<Eigen::Vector2i,
                                                        1>()))),
            make_tuple_iterator(
                    thrust::make_permutation_iterator(
                            source.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.end(),
                                    element_get_functor<Eigen::Vector2i, 0>())),
                    thrust::make_permutation_iterator(
                            target.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.end(),
                                    element_get_functor<Eigen::Vector2i, 1>())),
                    thrust::make_permutation_iterator(
                            target.normals_.begin(),
                            thrust::make_transform_iterator(
                                    corres.end(),
                                    element_get_functor<Eigen::Vector2i,
                                                        1>()))),
            [] __device__(const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f,
                                              Eigen::Vector3f> &x) -> float {
                float r = (thrust::get<0>(x) - thrust::get<1>(x))
                                  .dot(thrust::get<2>(x));
                return r * r;
            },
            0.0f, thrust::plus<float>());
    return std::sqrt(err / (float)corres.size());
}

Eigen::Matrix4f TransformationEstimationPointToPlane::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || !target.HasNormals())
        return Eigen::Matrix4f::Identity();

    Eigen::Matrix6f JTJ;
    Eigen::Vector6f JTr;
    float r2;
    pt2pl_jacobian_residual_functor func(
            thrust::raw_pointer_cast(source.points_.data()),
            thrust::raw_pointer_cast(target.points_.data()),
            thrust::raw_pointer_cast(target.normals_.data()),
            thrust::raw_pointer_cast(corres.data()));
    thrust::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix6f, Eigen::Vector6f,
                                      pt2pl_jacobian_residual_functor>(
                    func, (int)corres.size());

    bool is_success;
    Eigen::Matrix4f extrinsic;
    thrust::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr,
                                                                 det_thresh_);

    return is_success ? extrinsic : Eigen::Matrix4f::Identity();
}

float TransformationEstimationSymmetricMethod::ComputeRMSE(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || !source.HasNormals() || !target.HasNormals())
        return 0.0;
    const float err = thrust::transform_reduce(
            utility::exec_policy(0),
            make_tuple_iterator(
                    thrust::make_permutation_iterator(
                            source.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.begin(),
                                    element_get_functor<Eigen::Vector2i, 0>())),
                    thrust::make_permutation_iterator(
                            source.normals_.begin(),
                            thrust::make_transform_iterator(
                                    corres.begin(),
                                    element_get_functor<Eigen::Vector2i, 0>())),
                    thrust::make_permutation_iterator(
                            target.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.begin(),
                                    element_get_functor<Eigen::Vector2i, 1>())),
                    thrust::make_permutation_iterator(
                            target.normals_.begin(),
                            thrust::make_transform_iterator(
                                    corres.begin(),
                                    element_get_functor<Eigen::Vector2i,
                                                        1>()))),
            make_tuple_iterator(
                    thrust::make_permutation_iterator(
                            source.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.end(),
                                    element_get_functor<Eigen::Vector2i, 0>())),
                    thrust::make_permutation_iterator(
                            source.normals_.begin(),
                            thrust::make_transform_iterator(
                                    corres.end(),
                                    element_get_functor<Eigen::Vector2i, 0>())),
                    thrust::make_permutation_iterator(
                            target.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.end(),
                                    element_get_functor<Eigen::Vector2i, 1>())),
                    thrust::make_permutation_iterator(
                            target.normals_.begin(),
                            thrust::make_transform_iterator(
                                    corres.end(),
                                    element_get_functor<Eigen::Vector2i,
                                                        1>()))),
            [] __device__(
                    const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f,
                                        Eigen::Vector3f, Eigen::Vector3f> &x) -> float{
                // Compute error using both source and target normals
                float r = ComputeErrorUsingNormals(
                        thrust::get<0>(x), thrust::get<1>(x), thrust::get<2>(x),
                        thrust::get<3>(x));
                return r * r;
            },
            0.0f, thrust::plus<float>());
    return std::sqrt(err / (float)corres.size());
}

Eigen::Matrix4f TransformationEstimationSymmetricMethod::ComputeTransformation(
        const geometry::PointCloud &source,
        const geometry::PointCloud &target,
        const CorrespondenceSet &corres) const {
    if (corres.empty() || !source.HasNormals() || !target.HasNormals())
        return Eigen::Matrix4f::Identity();

    Eigen::Matrix6f JTJ;
    Eigen::Vector6f JTr;
    float r2;
    symmetric_jacobian_residual_functor func(
            thrust::raw_pointer_cast(source.points_.data()),
            thrust::raw_pointer_cast(source.normals_.data()),
            thrust::raw_pointer_cast(target.points_.data()),
            thrust::raw_pointer_cast(target.normals_.data()),
            thrust::raw_pointer_cast(corres.data()));
    thrust::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix6f, Eigen::Vector6f,
                                      symmetric_jacobian_residual_functor>(
                    func, (int)corres.size());

    bool is_success;
    Eigen::Matrix4f transformation_matrix;
    thrust::tie(is_success, transformation_matrix) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr,
                                                                 det_thresh_);
    if (is_success) {
        // Extract the rotation matrix (3x3 upper-left block) from the 4x4
        // transformation matrix. This matrix represents the rotation component
        // of the transformation.
        Eigen::Matrix3f rotation_matrix_3f =
                transformation_matrix.template block<3, 3>(0, 0);

        // Convert the rotation matrix to double precision for higher accuracy.
        // This step is essential to maintain accuracy in the subsequent
        // squaring operation.
        Eigen::Matrix3d rotation_matrix_3d = rotation_matrix_3f.cast<double>();

        // Square the rotation matrix as described in the original Symmetric ICP
        // paper. This approach, part of the rotated-normals version of the
        // symmetric objective, optimizes for half of the rotation angle,
        // reducing linearization error and enabling exact correspondences in
        // certain cases.
        Eigen::Matrix3d R = rotation_matrix_3d * rotation_matrix_3d;

        // Construct the final transformation matrix using the squared rotation
        // matrix (R) and the translation vector. The translation component is
        // extracted directly from the original 4x4 transformation matrix.
        Eigen::Matrix4f extrinsic = Eigen::Matrix4f::Identity();
        extrinsic.template block<3, 3>(0, 0) =
                R.cast<float>();  // Set the rotation part with the squared
                                  // rotation matrix.
        extrinsic.template block<3, 1>(0, 3) =
                transformation_matrix.template block<3, 1>(
                        0, 3);  // Set the translation part from the original
                                // transformation matrix.

        return extrinsic;
    }

    return Eigen::Matrix4f::Identity();
}