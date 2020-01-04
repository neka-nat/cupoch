#include "cupoch/utility/eigen.h"
#include "cupoch/utility/helper.h"
#include "cupoch/utility/console.h"
#include <Eigen/Geometry>
#include <thrust/transform_reduce.h>

using namespace cupoch;
using namespace cupoch::utility;

namespace {

template<typename MatType, typename VecType>
struct jtj_jtr_reduce_functor {
    jtj_jtr_reduce_functor(jacobian_residual_functor<VecType>& f) : f_(f) {};
    const jacobian_residual_functor<VecType>& f_;
    __device__
    thrust::tuple<MatType, VecType, float> operator() (int idx) const {
        VecType J_r;
        float r;
        f_(idx, J_r, r);
        MatType jtj = J_r * J_r.transpose();
        VecType jr = J_r * r;
        return thrust::make_tuple(jtj, jr, r * r);
    }
};

}

Eigen::Matrix4f cupoch::utility::TransformVector6fToMatrix4f(const Eigen::Vector6f &input) {
    Eigen::Matrix4f output = Eigen::Matrix4f::Identity();
    output.block<3, 3>(0, 0) =
            (Eigen::AngleAxisf(input(2), Eigen::Vector3f::UnitZ()) *
             Eigen::AngleAxisf(input(1), Eigen::Vector3f::UnitY()) *
             Eigen::AngleAxisf(input(0), Eigen::Vector3f::UnitX()))
                    .matrix();
    output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
    return output;
}

template<int Dim>
thrust::tuple<bool, Eigen::Matrix<float, Dim, 1>> cupoch::utility::SolveLinearSystemPSD(
        const Eigen::Matrix<float, Dim, Dim> &A,
        const Eigen::Matrix<float, Dim, 1> &b,
        bool check_symmetric,
        bool check_det) {
    // PSD implies symmetric
    if (check_symmetric && !A.isApprox(A.transpose())) {
        LogWarning("check_symmetric failed, empty vector will be returned");
        return thrust::make_tuple(false, Eigen::Matrix<float, Dim, 1>::Zero());
    }

    if (check_det) {
        float det = A.determinant();
        if (fabs(det) < 1e-6 || std::isnan(det) || std::isinf(det)) {
            LogWarning("check_det failed, empty vector will be returned");
            return thrust::make_tuple(false, Eigen::Matrix<float, Dim, 1>::Zero());
        }
    }

    Eigen::Matrix<float, Dim, 1> x;

    x = A.ldlt().solve(b);
    return thrust::make_tuple(true, std::move(x));
}

thrust::tuple<bool, Eigen::Matrix4f> cupoch::utility::SolveJacobianSystemAndObtainExtrinsicMatrix(
    const Eigen::Matrix6f &JTJ, const Eigen::Vector6f &JTr) {
    bool solution_exist;
    Eigen::Vector6f x;
    thrust::tie(solution_exist, x) = SolveLinearSystemPSD(JTJ, Eigen::Vector6f(-JTr));

    if (solution_exist) {
        Eigen::Matrix4f extrinsic = TransformVector6fToMatrix4f(x);
        return thrust::make_tuple(solution_exist, std::move(extrinsic));
    } else {
        return thrust::make_tuple(false, Eigen::Matrix4f::Identity());
    }
}

template <typename MatType, typename VecType>
thrust::tuple<MatType, VecType, float> cupoch::utility::ComputeJTJandJTr(
        jacobian_residual_functor<VecType>& f,
        int iteration_num,
        bool verbose) {
    MatType JTJ;
    VecType JTr;
    float r2_sum = 0.0;
    JTJ.setZero();
    JTr.setZero();
    jtj_jtr_reduce_functor<Eigen::Matrix6f, Eigen::Vector6f> func(f);
    auto jtj_jtr_r2 = thrust::transform_reduce(thrust::make_counting_iterator(0),
                                               thrust::make_counting_iterator(iteration_num),
                                               func, thrust::make_tuple(JTJ, JTr, r2_sum),
                                               thrust::plus<thrust::tuple<MatType, VecType, float>>());
    r2_sum = thrust::get<2>(jtj_jtr_r2);
    if (verbose) {
        LogDebug("Residual : {:.2e} (# of elements : {:d})",
                 r2_sum / (float)iteration_num, iteration_num);
    }
    return jtj_jtr_r2;
}

template thrust::tuple<bool, Eigen::Matrix<float, 6, 1>> cupoch::utility::SolveLinearSystemPSD(
    const Eigen::Matrix<float, 6, 6> &A,
    const Eigen::Matrix<float, 6, 1> &b,
    bool check_symmetric,
    bool check_det);

template thrust::tuple<Eigen::Matrix6f, Eigen::Vector6f, float> cupoch::utility::ComputeJTJandJTr(
    jacobian_residual_functor<Eigen::Vector6f>& f,
    int iteration_num, bool verbose);

Eigen::Matrix3f cupoch::utility::RotationMatrixX(float radians) {
    Eigen::Matrix3f rot;
    rot << 1, 0, 0, 0, std::cos(radians), -std::sin(radians), 0,
            std::sin(radians), std::cos(radians);
    return rot;
}

Eigen::Matrix3f cupoch::utility::RotationMatrixY(float radians) {
    Eigen::Matrix3f rot;
    rot << std::cos(radians), 0, std::sin(radians), 0, 1, 0, -std::sin(radians),
            0, std::cos(radians);
    return rot;
}

Eigen::Matrix3f cupoch::utility::RotationMatrixZ(float radians) {
    Eigen::Matrix3f rot;
    rot << std::cos(radians), -std::sin(radians), 0, std::sin(radians),
            std::cos(radians), 0, 0, 0, 1;
    return rot;
}