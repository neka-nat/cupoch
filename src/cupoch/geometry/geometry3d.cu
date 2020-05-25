#include <Eigen/Dense>

#include "cupoch/geometry/geometry3d.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct transform_points_functor {
    transform_points_functor(const Eigen::Matrix4f &transform)
        : transform_(transform){};
    const Eigen::Matrix4f transform_;
    __device__ void operator()(Eigen::Vector3f &pt) {
        const Eigen::Vector4f new_pt =
                transform_ * Eigen::Vector4f(pt(0), pt(1), pt(2), 1.0);
        pt = new_pt.head<3>() / new_pt(3);
    }
};

struct transform_normals_functor {
    transform_normals_functor(const Eigen::Matrix4f &transform)
        : transform_(transform){};
    const Eigen::Matrix4f transform_;
    __device__ void operator()(Eigen::Vector3f &nl) {
        const Eigen::Vector4f new_pt =
                transform_ * Eigen::Vector4f(nl(0), nl(1), nl(2), 0.0);
        nl = new_pt.head<3>();
    }
};
}  // namespace

Eigen::Vector3f Geometry3D::ComputeMinBound(
        const utility::device_vector<Eigen::Vector3f> &points) const {
    return ComputeMinBound(0, points);
}

Eigen::Vector3f Geometry3D::ComputeMinBound(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Vector3f> &points) const {
    if (points.empty()) return Eigen::Vector3f::Zero();
    Eigen::Vector3f init = points[0];
    return thrust::reduce(utility::exec_policy(stream)->on(stream),
                          points.begin(), points.end(), init,
                          thrust::elementwise_minimum<Eigen::Vector3f>());
}

Eigen::Vector3f Geometry3D::ComputeMaxBound(
        const utility::device_vector<Eigen::Vector3f> &points) const {
    return ComputeMaxBound(0, points);
}

Eigen::Vector3f Geometry3D::ComputeMaxBound(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Vector3f> &points) const {
    if (points.empty()) return Eigen::Vector3f::Zero();
    Eigen::Vector3f init = points[0];
    return thrust::reduce(utility::exec_policy(stream)->on(stream),
                          points.begin(), points.end(), init,
                          thrust::elementwise_maximum<Eigen::Vector3f>());
}

Eigen::Vector3f Geometry3D::ComputeCenter(
        const utility::device_vector<Eigen::Vector3f> &points) const {
    Eigen::Vector3f init = Eigen::Vector3f::Zero();
    if (points.empty()) return init;
    Eigen::Vector3f sum = thrust::reduce(points.begin(), points.end(), init,
                                         thrust::plus<Eigen::Vector3f>());
    return sum / points.size();
}

void Geometry3D::ResizeAndPaintUniformColor(
        utility::device_vector<Eigen::Vector3f> &colors,
        const size_t size,
        const Eigen::Vector3f &color) {
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

void Geometry3D::TransformPoints(
        const Eigen::Matrix4f &transformation,
        utility::device_vector<Eigen::Vector3f> &points) {
    TransformPoints(0, transformation, points);
}

void Geometry3D::TransformPoints(
        cudaStream_t stream,
        const Eigen::Matrix4f &transformation,
        utility::device_vector<Eigen::Vector3f> &points) {
    transform_points_functor func(transformation);
    thrust::for_each(utility::exec_policy(stream)->on(stream), points.begin(),
                     points.end(), func);
}

void Geometry3D::TransformNormals(
        const Eigen::Matrix4f &transformation,
        utility::device_vector<Eigen::Vector3f> &normals) {
    TransformNormals(0, transformation, normals);
}

void Geometry3D::TransformNormals(
        cudaStream_t stream,
        const Eigen::Matrix4f &transformation,
        utility::device_vector<Eigen::Vector3f> &normals) {
    transform_normals_functor func(transformation);
    thrust::for_each(utility::exec_policy(stream)->on(stream), normals.begin(),
                     normals.end(), func);
}

void Geometry3D::TranslatePoints(
        const Eigen::Vector3f &translation,
        utility::device_vector<Eigen::Vector3f> &points,
        bool relative) const {
    Eigen::Vector3f transform = translation;
    if (!relative) {
        transform -= ComputeCenter(points);
    }
    thrust::for_each(points.begin(), points.end(),
                     [=] __device__(Eigen::Vector3f & pt) { pt += transform; });
}

void Geometry3D::ScalePoints(const float scale,
                             utility::device_vector<Eigen::Vector3f> &points,
                             bool center) const {
    Eigen::Vector3f points_center(0, 0, 0);
    if (center && !points.empty()) {
        points_center = ComputeCenter(points);
    }
    thrust::for_each(points.begin(), points.end(),
                     [=] __device__(Eigen::Vector3f & pt) {
                         pt = (pt - points_center) * scale + points_center;
                     });
}

void Geometry3D::RotatePoints(const Eigen::Matrix3f &R,
                              utility::device_vector<Eigen::Vector3f> &points,
                              bool center) const {
    RotatePoints(0, R, points, center);
}

void Geometry3D::RotatePoints(cudaStream_t stream,
                              const Eigen::Matrix3f &R,
                              utility::device_vector<Eigen::Vector3f> &points,
                              bool center) const {
    Eigen::Vector3f points_center(0, 0, 0);
    if (center && !points.empty()) {
        points_center = ComputeCenter(points);
    }
    thrust::for_each(utility::exec_policy(stream)->on(stream), points.begin(),
                     points.end(), [=] __device__(Eigen::Vector3f & pt) {
                         pt = R * (pt - points_center) + points_center;
                     });
}

void Geometry3D::RotateNormals(
        const Eigen::Matrix3f &R,
        utility::device_vector<Eigen::Vector3f> &normals) const {
    RotateNormals(0, R, normals);
}

void Geometry3D::RotateNormals(
        cudaStream_t stream,
        const Eigen::Matrix3f &R,
        utility::device_vector<Eigen::Vector3f> &normals) const {
    thrust::for_each(utility::exec_policy(stream)->on(stream), normals.begin(),
                     normals.end(), [=] __device__(Eigen::Vector3f & normal) {
                         normal = R * normal;
                     });
}

Eigen::Matrix3f Geometry3D::GetRotationMatrixFromXYZ(
        const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixX(rotation(0)) *
           cupoch::utility::RotationMatrixY(rotation(1)) *
           cupoch::utility::RotationMatrixZ(rotation(2));
}

Eigen::Matrix3f Geometry3D::GetRotationMatrixFromYZX(
        const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixY(rotation(0)) *
           cupoch::utility::RotationMatrixZ(rotation(1)) *
           cupoch::utility::RotationMatrixX(rotation(2));
}

Eigen::Matrix3f Geometry3D::GetRotationMatrixFromZXY(
        const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixZ(rotation(0)) *
           cupoch::utility::RotationMatrixX(rotation(1)) *
           cupoch::utility::RotationMatrixY(rotation(2));
}

Eigen::Matrix3f Geometry3D::GetRotationMatrixFromXZY(
        const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixX(rotation(0)) *
           cupoch::utility::RotationMatrixZ(rotation(1)) *
           cupoch::utility::RotationMatrixY(rotation(2));
}

Eigen::Matrix3f Geometry3D::GetRotationMatrixFromZYX(
        const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixZ(rotation(0)) *
           cupoch::utility::RotationMatrixY(rotation(1)) *
           cupoch::utility::RotationMatrixX(rotation(2));
}

Eigen::Matrix3f Geometry3D::GetRotationMatrixFromYXZ(
        const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixY(rotation(0)) *
           cupoch::utility::RotationMatrixX(rotation(1)) *
           cupoch::utility::RotationMatrixZ(rotation(2));
}

Eigen::Matrix3f Geometry3D::GetRotationMatrixFromAxisAngle(
        const Eigen::Vector3f &rotation) {
    const float phi = rotation.norm();
    return Eigen::AngleAxisf(phi, rotation / phi).toRotationMatrix();
}

Eigen::Matrix3f Geometry3D::GetRotationMatrixFromQuaternion(
        const Eigen::Vector4f &rotation) {
    return Eigen::Quaternionf(rotation(0), rotation(1), rotation(2),
                              rotation(3))
            .normalized()
            .toRotationMatrix();
}