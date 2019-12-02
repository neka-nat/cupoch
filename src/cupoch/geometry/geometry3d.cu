#include "cupoch/geometry/geometry3d.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::geometry;

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

struct transform_points_functor {
    transform_points_functor(const Eigen::Matrix4f& transform) : transform_(transform){};
    const Eigen::Matrix4f transform_;
    __device__
    void operator()(Eigen::Vector3f_u& pt) {
        const Eigen::Vector4f new_pt = transform_ * Eigen::Vector4f(pt(0), pt(1), pt(2), 1.0);
        pt = new_pt.head<3>() / new_pt(3);
    }
};

struct transform_normals_functor {
    transform_normals_functor(const Eigen::Matrix4f& transform) : transform_(transform){};
    const Eigen::Matrix4f transform_;
    __device__
    void operator()(Eigen::Vector3f_u& nl) {
        const Eigen::Vector4f new_pt = transform_ * Eigen::Vector4f(nl(0), nl(1), nl(2), 0.0);
        nl = new_pt.head<3>();
    }
};
}

Eigen::Vector3f_u cupoch::geometry::ComputeMinBound(const thrust::device_vector<Eigen::Vector3f_u>& points) {
    if (points.empty()) return Eigen::Vector3f_u::Zero();
    Eigen::Vector3f_u init = points[0];
    return thrust::reduce(points.begin(), points.end(), init, elementwise_min_functor());
}

Eigen::Vector3f_u cupoch::geometry::ComputeMaxBound(const thrust::device_vector<Eigen::Vector3f_u>& points) {
    if (points.empty()) return Eigen::Vector3f_u::Zero();
    Eigen::Vector3f_u init = points[0];
    return thrust::reduce(points.begin(), points.end(), init, elementwise_max_functor());
}

Eigen::Vector3f_u cupoch::geometry::ComuteCenter(const thrust::device_vector<Eigen::Vector3f_u>& points) {
    Eigen::Vector3f_u init = Eigen::Vector3f_u::Zero();
    if (points.empty()) return init;
    Eigen::Vector3f_u sum = thrust::reduce(points.begin(), points.end(), init, thrust::plus<Eigen::Vector3f_u>());
    return sum / points.size();
}

void cupoch::geometry::ResizeAndPaintUniformColor(thrust::device_vector<Eigen::Vector3f_u>& colors,
    const size_t size,
    const Eigen::Vector3f& color) {
    colors.resize(size);
    Eigen::Vector3f clipped_color = color;
    if (color.minCoeff() < 0 || color.maxCoeff() > 1) {
        utility::LogWarning(
                "invalid color in PaintUniformColor, clipping to [0, 1]");
        clipped_color = clipped_color.array()
                                .max(Eigen::Vector3f(0, 0, 0).array())
                                .matrix();
        clipped_color = clipped_color.array()
                                .min(Eigen::Vector3f(1, 1, 1).array())
                                .matrix();
    }
    thrust::fill(colors.begin(), colors.end(), clipped_color);
}

void cupoch::geometry::TransformPoints(const Eigen::Matrix4f& transformation,
                                      thrust::device_vector<Eigen::Vector3f_u>& points) {
    transform_points_functor func(transformation);
    thrust::for_each(points.begin(), points.end(), func);
}

void cupoch::geometry::TransformNormals(const Eigen::Matrix4f& transformation,
                                      thrust::device_vector<Eigen::Vector3f_u>& normals) {
    transform_normals_functor func(transformation);
    thrust::for_each(normals.begin(), normals.end(), func);
}