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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

#include "cupoch/utility/console.h"
#include "cupoch/utility/eigen.h"
#include "cupoch/utility/helper.h"

namespace cupoch {
namespace utility {

namespace {

template <typename MatType, typename VecType, typename FuncType>
struct jtj_jtr_functor {
    jtj_jtr_functor(const FuncType &f) : f_(f){};
    const FuncType f_;
    __device__ thrust::tuple<MatType, VecType, float> operator()(
            int idx) const {
        VecType J_r;
        float r;
        f_(idx, J_r, r);
        MatType jtj = J_r * J_r.transpose();
        VecType jr = J_r * r;
        return thrust::make_tuple(jtj, jr, r * r);
    }
};

template <typename MatType, typename VecType, int NumJ, typename FuncType>
struct multiple_jtj_jtr_functor {
    multiple_jtj_jtr_functor(const FuncType &f) : f_(f){};
    const FuncType f_;
    __device__ thrust::tuple<MatType, VecType, float> operator()(
            int idx) const {
        MatType JTJ_private;
        VecType JTr_private;
        float r2_sum_private = 0.0;
        JTJ_private.setZero();
        JTr_private.setZero();
        VecType J_r[NumJ];
        float r[NumJ];
        f_(idx, J_r, r);
#pragma unroll
        for (size_t j = 0; j < NumJ; ++j) {
            JTJ_private.noalias() += J_r[j] * J_r[j].transpose();
            JTr_private.noalias() += J_r[j] * r[j];
            r2_sum_private += r[j] * r[j];
        }
        return thrust::make_tuple(JTJ_private, JTr_private, r2_sum_private);
    }
};

template <typename FuncType>
struct wrapped_calc_weights_functor {
    wrapped_calc_weights_functor(const FuncType &f, float r2_sum)
        : f_(f), r2_sum_(r2_sum){};
    const FuncType f_;
    const float r2_sum_;
    __device__ float operator()(float r2) const { return f_(r2, r2_sum_); };
};

}  // namespace


template <typename MatType, typename VecType, typename FuncType>
thrust::tuple<MatType, VecType, float> ComputeJTJandJTr(const FuncType &f,
                                                        int iteration_num,
                                                        bool verbose) {
    return ComputeJTJandJTr<MatType, VecType, FuncType>(
            0, f, iteration_num, verbose);
}

template <typename MatType, typename VecType, typename FuncType>
thrust::tuple<MatType, VecType, float> ComputeJTJandJTr(cudaStream_t stream,
                                                        const FuncType &f,
                                                        int iteration_num,
                                                        bool verbose) {
    MatType JTJ;
    VecType JTr;
    float r2_sum = 0.0;
    JTJ.setZero();
    JTr.setZero();
    jtj_jtr_functor<MatType, VecType, FuncType> func(f);
    auto jtj_jtr_r2 = thrust::transform_reduce(
            utility::exec_policy(stream),
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(iteration_num), func,
            thrust::make_tuple(JTJ, JTr, r2_sum),
            add_tuple_functor<MatType, VecType, float>());
    r2_sum = thrust::get<2>(jtj_jtr_r2);
    if (verbose) {
        LogDebug("Residual : {:.2e} (# of elements : {:d})",
                 r2_sum / (float)iteration_num, iteration_num);
    }
    return jtj_jtr_r2;
}

template <typename MatType, typename VecType, int NumJ, typename FuncType>
thrust::tuple<MatType, VecType, float> ComputeJTJandJTr(
        const FuncType &f, int iteration_num, bool verbose /*=true*/) {
    return ComputeJTJandJTr<MatType, VecType, NumJ, FuncType>(
            0, f, iteration_num, verbose);
}

template <typename MatType, typename VecType, int NumJ, typename FuncType>
thrust::tuple<MatType, VecType, float> ComputeJTJandJTr(
        cudaStream_t stream, const FuncType &f, int iteration_num, bool verbose /*=true*/) {
    MatType JTJ;
    VecType JTr;
    float r2_sum = 0.0;
    JTJ.setZero();
    JTr.setZero();
    multiple_jtj_jtr_functor<MatType, VecType, NumJ, FuncType> func(f);
    auto jtj_jtr_r2 = thrust::transform_reduce(
            utility::exec_policy(stream),
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(iteration_num), func,
            thrust::make_tuple(JTJ, JTr, r2_sum),
            add_tuple_functor<MatType, VecType, float>());
    r2_sum = thrust::get<2>(jtj_jtr_r2);
    if (verbose) {
        LogDebug("Residual : {:.2e} (# of elements : {:d})",
                 r2_sum / (float)iteration_num, iteration_num);
    }
    return jtj_jtr_r2;
}

template <typename MatType,
          typename VecType,
          int NumJ,
          typename FuncJType,
          typename FuncW1Type,
          typename FuncW2Type>
thrust::tuple<MatType, VecType, float, float> ComputeWeightedJTJandJTr(
        const FuncJType &fj,
        const FuncW1Type &fw_reduce,
        const FuncW2Type &fw_trans,
        const int iteration_num,
        bool verbose /*=true*/) {
    MatType JTJ;
    VecType JTr;
    float r2_sum = 0.0;
    JTJ.setZero();
    JTr.setZero();
    utility::device_vector<MatType> JTJs(iteration_num);
    utility::device_vector<VecType> JTrs(iteration_num);
    utility::device_vector<float> r2s(iteration_num);
    utility::device_vector<float> ws(iteration_num);
    multiple_jtj_jtr_functor<MatType, VecType, NumJ, FuncJType> funcj(fj);
    thrust::transform(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(iteration_num),
                      make_tuple_begin(JTJs, JTrs, r2s), funcj);
    float w_sum = thrust::transform_reduce(r2s.begin(), r2s.end(), fw_reduce,
                                           0.0, thrust::plus<float>());
    wrapped_calc_weights_functor<FuncW2Type> funcw(fw_trans, w_sum);
    thrust::transform(r2s.begin(), r2s.end(), ws.begin(), funcw);
    auto jtj_jtr_r2 = thrust::transform_reduce(
            make_tuple_begin(JTJs, JTrs, r2s, ws),
            make_tuple_end(JTJs, JTrs, r2s, ws),
            [] __device__(
                    const thrust::tuple<MatType, VecType, float, float> &x) -> thrust::tuple<MatType, VecType, float> {
                float w = thrust::get<3>(x);
                return thrust::make_tuple(thrust::get<0>(x) * w,
                                          thrust::get<1>(x) * w,
                                          thrust::get<2>(x) * w);
            },
            thrust::make_tuple(JTJ, JTr, r2_sum),
            add_tuple_functor<MatType, VecType, float>());
    if (verbose) {
        LogDebug("Residual : {:.2e} (# of elements : {:d})",
                 r2_sum / (float)iteration_num, iteration_num);
    }
    return thrust::make_tuple(thrust::get<0>(jtj_jtr_r2),
                              thrust::get<1>(jtj_jtr_r2),
                              thrust::get<2>(jtj_jtr_r2), w_sum);
}

template <int Dim, typename T, typename FuncT>
Eigen::Matrix<T, Dim, 1> ComputeBound(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Matrix<T, Dim, 1>> &points) {
    if (points.empty()) return Eigen::Matrix<T, Dim, 1>::Zero();
    Eigen::Matrix<T, Dim, 1> init = points[0];
    return thrust::reduce(utility::exec_policy(stream),
                          points.begin(), points.end(), init, FuncT());
}

template <int Dim, typename T>
Eigen::Matrix<T, Dim, 1> ComputeMinBound(
        const utility::device_vector<Eigen::Matrix<T, Dim, 1>> &points) {
    return ComputeBound<
            Dim, T, thrust::elementwise_minimum<Eigen::Matrix<T, Dim, 1>>>(
            0, points);
}

template <int Dim, typename T>
Eigen::Matrix<T, Dim, 1> ComputeMaxBound(
        const utility::device_vector<Eigen::Matrix<T, Dim, 1>> &points) {
    return ComputeBound<
            Dim, T, thrust::elementwise_maximum<Eigen::Matrix<T, Dim, 1>>>(
            0, points);
}

template <int Dim, typename T>
Eigen::Matrix<T, Dim, 1> ComputeCenter(
        const utility::device_vector<Eigen::Matrix<T, Dim, 1>> &points) {
    Eigen::Matrix<T, Dim, 1> init = Eigen::Matrix<T, Dim, 1>::Zero();
    if (points.empty()) return init;
    Eigen::Matrix<T, Dim, 1> sum = thrust::reduce(
            utility::exec_policy(0), points.begin(), points.end(), init,
            thrust::plus<Eigen::Matrix<float, Dim, 1>>());
    return sum / points.size();
}

}  // namespace utility
}  // namespace cupoch