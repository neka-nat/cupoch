#include "cupoch/registration/kabsch.h"
#include "cupoch/utility/svd3_cuda.h"
#include "cupoch/utility/global_stream.h"
#include <Eigen/Geometry>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

using namespace cupoch;
using namespace cupoch::registration;

namespace {

template<int Index>
struct extract_correspondence_functor {
    extract_correspondence_functor(const Eigen::Vector3f* points, const Eigen::Vector2i* corres)
        : points_(points), corres_(corres) {};
    const Eigen::Vector3f* points_;
    const Eigen::Vector2i* corres_;
    __device__
    Eigen::Vector3f operator() (size_t idx) const {
        return points_[corres_[idx][Index]];
    }
};

struct outer_product_functor {
    outer_product_functor(const Eigen::Vector3f* source, const Eigen::Vector3f* target,
                          const Eigen::Vector2i* corres, const Eigen::Vector3f& x_offset,
                          const Eigen::Vector3f& y_offset)
        : source_(source), target_(target), corres_(corres), x_offset_(x_offset), y_offset_(y_offset) {};
    const Eigen::Vector3f* source_;
    const Eigen::Vector3f* target_;
    const Eigen::Vector2i* corres_;
    const Eigen::Vector3f x_offset_;
    const Eigen::Vector3f y_offset_;
    __device__
    Eigen::Matrix3f operator() (size_t idx) const {
        const Eigen::Vector3f centralized_x = source_[corres_[idx][0]] - x_offset_;
        const Eigen::Vector3f centralized_y = target_[corres_[idx][1]] - y_offset_;
        Eigen::Matrix3f ans = centralized_x * centralized_y.transpose();
        return ans;
    }
};

struct set_correspondence_functor {
    __device__
    Eigen::Vector2i operator() (size_t idx) {
        return Eigen::Vector2i(idx, idx);
    }
};

}

Eigen::Matrix4f_u cupoch::registration::Kabsch(const thrust::device_vector<Eigen::Vector3f>& model,
                                               const thrust::device_vector<Eigen::Vector3f>& target,
                                               const CorrespondenceSet& corres) {
    //Compute the center
    extract_correspondence_functor<0> ex_func0(thrust::raw_pointer_cast(model.data()),
                                               thrust::raw_pointer_cast(corres.data()));
    extract_correspondence_functor<1> ex_func1(thrust::raw_pointer_cast(target.data()),
                                               thrust::raw_pointer_cast(corres.data()));
    Eigen::Vector3f model_center = thrust::transform_reduce(thrust::cuda::par.on(utility::GetGlobalStream(0)),
                                                            thrust::make_counting_iterator<size_t>(0),
                                                            thrust::make_counting_iterator(corres.size()),
                                                            ex_func0, Eigen::Vector3f(0.0, 0.0, 0.0),
                                                            thrust::plus<Eigen::Vector3f>());
    Eigen::Vector3f target_center = thrust::transform_reduce(thrust::cuda::par.on(utility::GetGlobalStream(1)),
                                                             thrust::make_counting_iterator<size_t>(0),
                                                             thrust::make_counting_iterator(corres.size()),
                                                             ex_func1, Eigen::Vector3f(0.0, 0.0, 0.0),
                                                             thrust::plus<Eigen::Vector3f>());
    cudaDeviceSynchronize();
    float divided_by = 1.0f / model.size();
    model_center *= divided_by;
    target_center *= divided_by;

    //Centralize them
    //Compute the H matrix
    outer_product_functor func(thrust::raw_pointer_cast(model.data()),
                               thrust::raw_pointer_cast(target.data()),
                               thrust::raw_pointer_cast(corres.data()),
                               model_center, target_center);
    const Eigen::Matrix3f init = Eigen::Matrix3f::Zero();
    Eigen::Matrix3f hh = thrust::transform_reduce(thrust::make_counting_iterator<size_t>(0),
                                                  thrust::make_counting_iterator(corres.size()),
                                                  func, init, thrust::plus<Eigen::Matrix3f>());

    //Do svd
    hh /= model.size();
    Eigen::Matrix3f uu, ss, vv;
    svd(hh(0, 0), hh(0, 1), hh(0, 2), hh(1, 0), hh(1, 1), hh(1, 2), hh(2, 0), hh(2, 1), hh(2, 2),
        uu(0, 0), uu(0, 1), uu(0, 2), uu(1, 0), uu(1, 1), uu(1, 2), uu(2, 0), uu(2, 1), uu(2, 2),
        ss(0, 0), ss(0, 1), ss(0, 2), ss(1, 0), ss(1, 1), ss(1, 2), ss(2, 0), ss(2, 1), ss(2, 2),
        vv(0, 0), vv(0, 1), vv(0, 2), vv(1, 0), vv(1, 1), vv(1, 2), vv(2, 0), vv(2, 1), vv(2, 2));
    ss = Eigen::Matrix3f::Identity();
    ss(2, 2) = (uu * vv).determinant();
    Eigen::Matrix4f_u tr = Eigen::Matrix4f_u::Identity();
    tr.block<3, 3>(0, 0) = vv * ss * uu.transpose();

    //The translation
    tr.block<3, 1>(0, 3) = target_center;
    tr.block<3, 1>(0, 3) -= tr.block<3, 3>(0, 0) * model_center;

    return tr;
}

Eigen::Matrix4f_u cupoch::registration::Kabsch(const thrust::device_vector<Eigen::Vector3f>& model,
                                               const thrust::device_vector<Eigen::Vector3f>& target) {
    CorrespondenceSet corres(model.size());
    set_correspondence_functor func;
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(model.size()), corres.begin(), func);
    return Kabsch(model, target, corres);
}