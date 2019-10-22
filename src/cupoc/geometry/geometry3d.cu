#include "cupoc/geometry/geometry3d.h"

using namespace cupoc;
using namespace cupoc::geometry;

namespace {

struct elementwise_min_functor {
    __device__
    Eigen::Vector3f operator()(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
        return a.array().min(b.array()).matrix();
    }
};
    
struct elementwise_max_functor {
    __device__
    Eigen::Vector3f operator()(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
        return a.array().max(b.array()).matrix();
    }
};

}

Eigen::Vector3f_u cupoc::geometry::ComputeMinBound(const thrust::device_vector<Eigen::Vector3f_u>& points) {
    if (points.empty()) return Eigen::Vector3f_u::Zero();
    Eigen::Vector3f_u init = points[0];
    return thrust::reduce(points.begin(), points.end(), init, elementwise_min_functor());
}

Eigen::Vector3f_u cupoc::geometry::ComputeMaxBound(const thrust::device_vector<Eigen::Vector3f_u>& points) {
    if (points.empty()) return Eigen::Vector3f_u::Zero();
    Eigen::Vector3f_u init = points[0];
    return thrust::reduce(points.begin(), points.end(), init, elementwise_max_functor());
}

Eigen::Vector3f_u cupoc::geometry::ComuteCenter(const thrust::device_vector<Eigen::Vector3f_u>& points) {
    Eigen::Vector3f_u init = Eigen::Vector3f_u::Zero();
    if (points.empty()) return init;
    Eigen::Vector3f_u sum = thrust::reduce(points.begin(), points.end(), init, thrust::plus<Eigen::Vector3f_u>());
    return sum / points.size();
}
