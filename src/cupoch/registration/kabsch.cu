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
#include <thrust/tabulate.h>
#include <thrust/async/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/permutation_iterator.h>

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "cupoch/registration/kabsch.h"
#include "cupoch/utility/platform.h"

using namespace cupoch;
using namespace cupoch::registration;

Eigen::Matrix4f_u cupoch::registration::Kabsch(const utility::device_vector<Eigen::Vector3f> &model,
                         const utility::device_vector<Eigen::Vector3f> &target,
                         const CorrespondenceSet &corres) {
    return Kabsch(utility::GetStream(0), utility::GetStream(1), model, target,
                  corres);
}

Eigen::Matrix4f_u cupoch::registration::Kabsch(
        cudaStream_t stream1, cudaStream_t stream2,
        const utility::device_vector<Eigen::Vector3f> &model,
        const utility::device_vector<Eigen::Vector3f> &target,
        const CorrespondenceSet &corres) {
    // Compute the center
    auto res1 = thrust::async::reduce(
            utility::exec_policy(stream1),
            thrust::make_permutation_iterator(
                    model.begin(),
                    thrust::make_transform_iterator(
                            corres.begin(),
                            element_get_functor<Eigen::Vector2i, 0>())),
            thrust::make_permutation_iterator(
                    model.begin(),
                    thrust::make_transform_iterator(
                            corres.end(),
                            element_get_functor<Eigen::Vector2i, 0>())),
            Eigen::Vector3f(0.0, 0.0, 0.0), thrust::plus<Eigen::Vector3f>());
    auto res2 = thrust::async::reduce(
            utility::exec_policy(stream2),
            thrust::make_permutation_iterator(
                    target.begin(),
                    thrust::make_transform_iterator(
                            corres.begin(),
                            element_get_functor<Eigen::Vector2i, 1>())),
            thrust::make_permutation_iterator(
                    target.begin(),
                    thrust::make_transform_iterator(
                            corres.end(),
                            element_get_functor<Eigen::Vector2i, 1>())),
            Eigen::Vector3f(0.0, 0.0, 0.0), thrust::plus<Eigen::Vector3f>());
    Eigen::Vector3f model_center = res1.get();
    Eigen::Vector3f target_center = res2.get();
    float divided_by = 1.0f / model.size();
    model_center *= divided_by;
    target_center *= divided_by;

    // Centralize them
    // Compute the H matrix
    const Eigen::Matrix3f init = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f hh = thrust::inner_product(
            utility::exec_policy(stream1),
            thrust::make_permutation_iterator(
                    model.begin(),
                    thrust::make_transform_iterator(
                            corres.begin(),
                            element_get_functor<Eigen::Vector2i, 0>())),
            thrust::make_permutation_iterator(
                    model.begin(),
                    thrust::make_transform_iterator(
                            corres.end(),
                            element_get_functor<Eigen::Vector2i, 0>())),
            thrust::make_permutation_iterator(
                    target.begin(),
                    thrust::make_transform_iterator(
                            corres.begin(),
                            element_get_functor<Eigen::Vector2i, 1>())),
            init, thrust::plus<Eigen::Matrix3f>(),
            [model_center, target_center] __device__(
                    const Eigen::Vector3f &lhs, const Eigen::Vector3f &rhs) {
                return (lhs - model_center) * (rhs - target_center).transpose();
            });

    // Do svd
    hh /= model.size();
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(
            hh, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f ss = Eigen::Matrix3f::Identity();
    ss(2, 2) = (svd.matrixU() * svd.matrixV()).determinant();
    Eigen::Matrix4f tr = Eigen::Matrix4f::Identity();
    tr.block<3, 3>(0, 0) = svd.matrixV() * ss * svd.matrixU().transpose();

    // The translation
    tr.block<3, 1>(0, 3) = target_center;
    tr.block<3, 1>(0, 3) -= tr.block<3, 3>(0, 0) * model_center;

    return tr;
}

Eigen::Matrix4f_u cupoch::registration::Kabsch(const utility::device_vector<Eigen::Vector3f> &model,
                         const utility::device_vector<Eigen::Vector3f> &target) {
    return Kabsch(utility::GetStream(0), utility::GetStream(1), model, target);
}

Eigen::Matrix4f_u cupoch::registration::Kabsch(
        cudaStream_t stream1, cudaStream_t stream2,
        const utility::device_vector<Eigen::Vector3f> &model,
        const utility::device_vector<Eigen::Vector3f> &target) {
    CorrespondenceSet corres(model.size());
    thrust::tabulate(corres.begin(), corres.end(), [] __device__(size_t idx) {
        return Eigen::Vector2i(idx, idx);
    });
    return Kabsch(stream1, stream2, model, target, corres);
}

Eigen::Matrix4f_u cupoch::registration::Kabsch(
        const std::vector<Eigen::Vector3f> &model,
        const std::vector<Eigen::Vector3f> &target) {
    return Kabsch(utility::device_vector<Eigen::Vector3f>(model),
                  utility::device_vector<Eigen::Vector3f>(target));
}

Eigen::Matrix4f_u cupoch::registration::KabschWeighted(
        const utility::device_vector<Eigen::Vector3f> &model,
        const utility::device_vector<Eigen::Vector3f> &target,
        const utility::device_vector<float> &weight) {
    // Compute the center
    auto res_w =
            thrust::async::reduce(utility::exec_policy(utility::GetStream(0)),
                                  weight.begin(), weight.end(), 0.0f);
    Eigen::Vector3f model_center = thrust::transform_reduce(
            utility::exec_policy(0), make_tuple_begin(model, weight),
            make_tuple_end(model, weight),
            [] __device__(const thrust::tuple<Eigen::Vector3f, float> &x) -> Eigen::Vector3f {
                return thrust::get<0>(x) * thrust::get<1>(x);
            },
            Eigen::Vector3f(0.0, 0.0, 0.0), thrust::plus<Eigen::Vector3f>());
    Eigen::Vector3f target_center = thrust::transform_reduce(
            utility::exec_policy(0), make_tuple_begin(target, weight),
            make_tuple_end(target, weight),
            [] __device__(const thrust::tuple<Eigen::Vector3f, float> &x) -> Eigen::Vector3f {
                return thrust::get<0>(x) * thrust::get<1>(x);
            },
            Eigen::Vector3f(0.0, 0.0, 0.0), thrust::plus<Eigen::Vector3f>());
    float total_weight = res_w.get();
    float divided_by = 1.0f / total_weight;
    model_center *= divided_by;
    target_center *= divided_by;

    // Centralize them
    // Compute the H matrix
    const float h_weight = thrust::transform_reduce(
            utility::exec_policy(0), weight.begin(), weight.end(),
            [] __device__(float x) -> float { return x * x; }, 0.0f,
            thrust::plus<float>());
    const Eigen::Matrix3f init = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f hh = thrust::transform_reduce(
            utility::exec_policy(0),
            make_tuple_begin(model, target, weight),
            make_tuple_end(model, target, weight),
            [model_center, target_center] __device__(
                    const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, float>
                            &x) -> Eigen::Matrix3f {
                const Eigen::Vector3f centralized_x =
                        thrust::get<0>(x) - model_center;
                const Eigen::Vector3f centralized_y =
                        thrust::get<1>(x) - target_center;
                const float w = thrust::get<2>(x);
                return w * w * centralized_x * centralized_y.transpose();
            },
            init, thrust::plus<Eigen::Matrix3f>());

    // Do svd
    hh /= h_weight;
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(
            hh, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f ss = Eigen::Matrix3f::Identity();
    ss(2, 2) = (svd.matrixU() * svd.matrixV()).determinant();
    Eigen::Matrix4f tr = Eigen::Matrix4f::Identity();
    tr.block<3, 3>(0, 0) = svd.matrixV() * ss * svd.matrixU().transpose();

    // The translation
    tr.block<3, 1>(0, 3) = target_center;
    tr.block<3, 1>(0, 3) -= tr.block<3, 3>(0, 0) * model_center;

    return tr;
}