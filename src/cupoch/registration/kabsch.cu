#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/permutation_iterator.h>

#include <Eigen/Geometry>

#include "cupoch/registration/kabsch.h"
#include "cupoch/utility/svd3_cuda.h"

using namespace cupoch;
using namespace cupoch::registration;

Eigen::Matrix4f_u cupoch::registration::Kabsch(
        const utility::device_vector<Eigen::Vector3f> &model,
        const utility::device_vector<Eigen::Vector3f> &target,
        const CorrespondenceSet &corres) {
    // Compute the center
    Eigen::Vector3f model_center = thrust::reduce(
            thrust::make_permutation_iterator(model.begin(),
                    thrust::make_transform_iterator(corres.begin(), extract_element_functor<int, 2, 0>())),
            thrust::make_permutation_iterator(model.begin(),
                    thrust::make_transform_iterator(corres.end(), extract_element_functor<int, 2, 0>())),
            Eigen::Vector3f(0.0, 0.0, 0.0), thrust::plus<Eigen::Vector3f>());
    Eigen::Vector3f target_center = thrust::reduce(
            thrust::make_permutation_iterator(target.begin(),
                    thrust::make_transform_iterator(corres.begin(), extract_element_functor<int, 2, 1>())),
            thrust::make_permutation_iterator(target.begin(),
                    thrust::make_transform_iterator(corres.end(), extract_element_functor<int, 2, 1>())),
            Eigen::Vector3f(0.0, 0.0, 0.0), thrust::plus<Eigen::Vector3f>());
    float divided_by = 1.0f / model.size();
    model_center *= divided_by;
    target_center *= divided_by;

    // Centralize them
    // Compute the H matrix
    const Eigen::Matrix3f init = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f hh = thrust::inner_product(
            thrust::make_permutation_iterator(model.begin(),
                    thrust::make_transform_iterator(corres.begin(), extract_element_functor<int, 2, 0>())),
            thrust::make_permutation_iterator(model.begin(),
                    thrust::make_transform_iterator(corres.end(), extract_element_functor<int, 2, 0>())),
            thrust::make_permutation_iterator(target.begin(),
                    thrust::make_transform_iterator(corres.begin(), extract_element_functor<int, 2, 1>())),
            init, thrust::plus<Eigen::Matrix3f>(),
            [model_center, target_center] __device__ (const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
                return (lhs - model_center) * (rhs - target_center).transpose();
            });

    // Do svd
    hh /= model.size();
    Eigen::Matrix3f uu, ss, vv;
    svd(hh(0, 0), hh(0, 1), hh(0, 2), hh(1, 0), hh(1, 1), hh(1, 2), hh(2, 0),
        hh(2, 1), hh(2, 2), uu(0, 0), uu(0, 1), uu(0, 2), uu(1, 0), uu(1, 1),
        uu(1, 2), uu(2, 0), uu(2, 1), uu(2, 2), ss(0, 0), ss(0, 1), ss(0, 2),
        ss(1, 0), ss(1, 1), ss(1, 2), ss(2, 0), ss(2, 1), ss(2, 2), vv(0, 0),
        vv(0, 1), vv(0, 2), vv(1, 0), vv(1, 1), vv(1, 2), vv(2, 0), vv(2, 1),
        vv(2, 2));
    ss = Eigen::Matrix3f::Identity();
    ss(2, 2) = (uu * vv).determinant();
    Eigen::Matrix4f_u tr = Eigen::Matrix4f_u::Identity();
    tr.block<3, 3>(0, 0) = vv * ss * uu.transpose();

    // The translation
    tr.block<3, 1>(0, 3) = target_center;
    tr.block<3, 1>(0, 3) -= tr.block<3, 3>(0, 0) * model_center;

    return tr;
}

Eigen::Matrix4f_u cupoch::registration::Kabsch(
        const utility::device_vector<Eigen::Vector3f> &model,
        const utility::device_vector<Eigen::Vector3f> &target) {
    CorrespondenceSet corres(model.size());
    thrust::tabulate(corres.begin(), corres.end(),
                     [] __device__ (size_t idx) {
                         return Eigen::Vector2i(idx, idx);
                     });
    return Kabsch(model, target, corres);
}