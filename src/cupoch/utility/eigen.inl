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
    wrapped_calc_weights_functor(const FuncType &f, float r2_sum) : f_(f), r2_sum_(r2_sum) {};
    const FuncType f_;
    const float r2_sum_;
    __device__ float operator() (float r2) const {
        return f_(r2, r2_sum_);
    };
};

}  // namespace

template <typename MatType, typename VecType, typename FuncType>
thrust::tuple<MatType, VecType, float> ComputeJTJandJTr(const FuncType &f,
                                                        int iteration_num,
                                                        bool verbose) {
    MatType JTJ;
    VecType JTr;
    float r2_sum = 0.0;
    JTJ.setZero();
    JTr.setZero();
    jtj_jtr_functor<MatType, VecType, FuncType> func(f);
    auto jtj_jtr_r2 = thrust::transform_reduce(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(iteration_num), func,
            thrust::make_tuple(JTJ, JTr, r2_sum),
            thrust::plus<thrust::tuple<MatType, VecType, float>>());
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
    MatType JTJ;
    VecType JTr;
    float r2_sum = 0.0;
    JTJ.setZero();
    JTr.setZero();
    multiple_jtj_jtr_functor<MatType, VecType, NumJ, FuncType> func(f);
    auto jtj_jtr_r2 = thrust::transform_reduce(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(iteration_num), func,
            thrust::make_tuple(JTJ, JTr, r2_sum),
            thrust::plus<thrust::tuple<MatType, VecType, float>>());
    r2_sum = thrust::get<2>(jtj_jtr_r2);
    if (verbose) {
        LogDebug("Residual : {:.2e} (# of elements : {:d})",
                 r2_sum / (float)iteration_num, iteration_num);
    }
    return jtj_jtr_r2;
}

template <typename MatType, typename VecType,
          int NumJ, typename FuncJType,
          typename FuncW1Type, typename FuncW2Type>
thrust::tuple<MatType, VecType, float, float> ComputeWeightedJTJandJTr(
        const FuncJType &fj, const FuncW1Type &fw_reduce,
        const FuncW2Type &fw_trans, const int iteration_num,
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
    thrust::transform(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(iteration_num),
            make_tuple_begin(JTJs, JTrs, r2s), funcj);
    float w_sum = thrust::transform_reduce(r2s.begin(), r2s.end(), fw_reduce, 0.0, thrust::plus<float>());
    wrapped_calc_weights_functor<FuncW2Type> funcw(fw_trans, w_sum);
    thrust::transform(r2s.begin(), r2s.end(), ws.begin(), funcw);
    auto jtj_jtr_r2 = thrust::transform_reduce(make_tuple_begin(JTJs, JTrs, r2s, ws),
                            make_tuple_end(JTJs, JTrs, r2s, ws),
                            [] __device__ (const thrust::tuple<MatType, VecType, float, float>& x) {
                                float w = thrust::get<3>(x);
                                return thrust::make_tuple(thrust::get<0>(x) * w,
                                                          thrust::get<1>(x) * w,
                                                          thrust::get<2>(x) * w);
                            },
                            thrust::make_tuple(JTJ, JTr, r2_sum),
                            thrust::plus<thrust::tuple<MatType, VecType, float>>());
    if (verbose) {
        LogDebug("Residual : {:.2e} (# of elements : {:d})",
                 r2_sum / (float)iteration_num, iteration_num);
    }
    return thrust::make_tuple(thrust::get<0>(jtj_jtr_r2),
                              thrust::get<1>(jtj_jtr_r2),
                              thrust::get<2>(jtj_jtr_r2),
                              w_sum);
}

}  // namespace utility
}  // namespace cupoch