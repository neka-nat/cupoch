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
                            extract_element_functor<int, 2, 0>())),
            thrust::make_permutation_iterator(
                    source.points_.begin(),
                    thrust::make_transform_iterator(
                            corres.end(),
                            extract_element_functor<int, 2, 0>())),
            thrust::make_permutation_iterator(
                    target.points_.begin(),
                    thrust::make_transform_iterator(
                            corres.begin(),
                            extract_element_functor<int, 2, 1>())),
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
            make_tuple_iterator(
                    thrust::make_permutation_iterator(
                            source.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.begin(),
                                    extract_element_functor<int, 2, 0>())),
                    thrust::make_permutation_iterator(
                            target.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.begin(),
                                    extract_element_functor<int, 2, 1>())),
                    thrust::make_permutation_iterator(
                            target.normals_.begin(),
                            thrust::make_transform_iterator(
                                    corres.begin(),
                                    extract_element_functor<int, 2, 1>()))),
            make_tuple_iterator(
                    thrust::make_permutation_iterator(
                            source.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.end(),
                                    extract_element_functor<int, 2, 0>())),
                    thrust::make_permutation_iterator(
                            target.points_.begin(),
                            thrust::make_transform_iterator(
                                    corres.end(),
                                    extract_element_functor<int, 2, 1>())),
                    thrust::make_permutation_iterator(
                            target.normals_.begin(),
                            thrust::make_transform_iterator(
                                    corres.end(),
                                    extract_element_functor<int, 2, 1>()))),
            [] __device__(const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f,
                                              Eigen::Vector3f> &x) {
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
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr, det_thresh_);

    return is_success ? extrinsic : Eigen::Matrix4f::Identity();
}