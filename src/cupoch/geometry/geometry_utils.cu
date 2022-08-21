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
#include <Eigen/Dense>

#include "cupoch/geometry/geometry_utils.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace geometry {

namespace {

template <int Dim>
struct transform_points_functor {
    transform_points_functor(
            const Eigen::Matrix<float, Dim + 1, Dim + 1> &transform)
        : transform_(transform){};
    const Eigen::Matrix<float, Dim + 1, Dim + 1> transform_;
    __device__ void operator()(Eigen::Matrix<float, Dim, 1> &pt) {
        pt = transform_.template block<Dim, Dim>(0, 0) * pt +
             transform_.template block<Dim, 1>(0, Dim);
    }
};

struct transform_normals_functor {
    transform_normals_functor(const Eigen::Matrix4f &transform)
        : transform_(transform){};
    const Eigen::Matrix4f transform_;
    __device__ void operator()(Eigen::Vector3f &nl) {
        nl = transform_.block<3, 3>(0, 0) * nl;
    }
};
}  // namespace

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
Eigen::Matrix<T, Dim, 1> ComputeMaxBound(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Matrix<T, Dim, 1>> &points) {
    if (points.empty()) return Eigen::Matrix<T, Dim, 1>::Zero();
    Eigen::Matrix<T, Dim, 1> init = points[0];
    return thrust::reduce(
            utility::exec_policy(stream), points.begin(),
            points.end(), init,
            thrust::elementwise_maximum<Eigen::Matrix<T, Dim, 1>>());
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

template Eigen::Matrix<float, 2, 1>
ComputeBound<2, float, thrust::elementwise_minimum<Eigen::Matrix<float, 2, 1>>>(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Matrix<float, 2, 1>> &points);
template Eigen::Matrix<float, 3, 1>
ComputeBound<3, float, thrust::elementwise_minimum<Eigen::Matrix<float, 3, 1>>>(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Matrix<float, 3, 1>> &points);
template Eigen::Matrix<float, 2, 1>
ComputeBound<2, float, thrust::elementwise_maximum<Eigen::Matrix<float, 2, 1>>>(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Matrix<float, 2, 1>> &points);
template Eigen::Matrix<float, 3, 1>
ComputeBound<3, float, thrust::elementwise_maximum<Eigen::Matrix<float, 3, 1>>>(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Matrix<float, 3, 1>> &points);

template Eigen::Matrix<float, 2, 1> ComputeMinBound<2, float>(
        const utility::device_vector<Eigen::Matrix<float, 2, 1>> &points);
template Eigen::Matrix<float, 3, 1> ComputeMinBound<3, float>(
        const utility::device_vector<Eigen::Matrix<float, 3, 1>> &points);
template Eigen::Matrix<int, 3, 1> ComputeMinBound<3, int>(
        const utility::device_vector<Eigen::Matrix<int, 3, 1>> &points);

template Eigen::Matrix<float, 2, 1> ComputeMaxBound<2, float>(
        const utility::device_vector<Eigen::Matrix<float, 2, 1>> &points);
template Eigen::Matrix<float, 3, 1> ComputeMaxBound<3, float>(
        const utility::device_vector<Eigen::Matrix<float, 3, 1>> &points);
template Eigen::Matrix<int, 3, 1> ComputeMaxBound<3, int>(
        const utility::device_vector<Eigen::Matrix<int, 3, 1>> &points);

template Eigen::Matrix<float, 2, 1> ComputeCenter<2, float>(
        const utility::device_vector<Eigen::Matrix<float, 2, 1>> &points);
template Eigen::Matrix<float, 3, 1> ComputeCenter<3, float>(
        const utility::device_vector<Eigen::Matrix<float, 3, 1>> &points);

void ResizeAndPaintUniformColor(utility::device_vector<Eigen::Vector3f> &colors,
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

template <int Dim>
void TransformPoints(
        const Eigen::Matrix<float, Dim + 1, Dim + 1> &transformation,
        utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points) {
    TransformPoints<Dim>(0, transformation, points);
}

template <int Dim>
void TransformPoints(
        cudaStream_t stream,
        const Eigen::Matrix<float, Dim + 1, Dim + 1> &transformation,
        utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points) {
    transform_points_functor<Dim> func(transformation);
    thrust::for_each(utility::exec_policy(stream), points.begin(),
                     points.end(), func);
}

template void TransformPoints<2>(
        const Eigen::Matrix3f &transformation,
        utility::device_vector<Eigen::Vector2f> &points);

template void TransformPoints<2>(
        cudaStream_t stream,
        const Eigen::Matrix3f &transformation,
        utility::device_vector<Eigen::Vector2f> &points);

template void TransformPoints<3>(
        const Eigen::Matrix4f &transformation,
        utility::device_vector<Eigen::Vector3f> &points);

template void TransformPoints<3>(
        cudaStream_t stream,
        const Eigen::Matrix4f &transformation,
        utility::device_vector<Eigen::Vector3f> &points);

void TransformNormals(const Eigen::Matrix4f &transformation,
                      utility::device_vector<Eigen::Vector3f> &normals) {
    TransformNormals(0, transformation, normals);
}

void TransformNormals(cudaStream_t stream,
                      const Eigen::Matrix4f &transformation,
                      utility::device_vector<Eigen::Vector3f> &normals) {
    transform_normals_functor func(transformation);
    thrust::for_each(utility::exec_policy(stream), normals.begin(),
                     normals.end(), func);
}

template <int Dim>
void TranslatePoints(
        const Eigen::Matrix<float, Dim, 1> &translation,
        utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
        bool relative) {
    Eigen::Matrix<float, Dim, 1> transform = translation;
    if (!relative) {
        transform -= ComputeCenter<Dim>(points);
    }
    thrust::for_each(points.begin(), points.end(),
                     [=] __device__(Eigen::Matrix<float, Dim, 1> & pt) {
                         pt += transform;
                     });
}

template <int Dim>
void ScalePoints(const float scale,
                 utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
                 bool center) {
    Eigen::Matrix<float, Dim, 1> points_center =
            Eigen::Matrix<float, Dim, 1>::Zero();
    if (center && !points.empty()) {
        points_center = ComputeCenter<Dim>(points);
    }
    thrust::for_each(points.begin(), points.end(),
                     [=] __device__(Eigen::Matrix<float, Dim, 1> & pt) {
                         pt = (pt - points_center) * scale + points_center;
                     });
}

template void TranslatePoints<2>(
        const Eigen::Vector2f &translation,
        utility::device_vector<Eigen::Vector2f> &points,
        bool relative);

template void TranslatePoints<3>(
        const Eigen::Vector3f &translation,
        utility::device_vector<Eigen::Vector3f> &points,
        bool relative);

template void ScalePoints<2>(const float scale,
                             utility::device_vector<Eigen::Vector2f> &points,
                             bool center);

template void ScalePoints<3>(const float scale,
                             utility::device_vector<Eigen::Vector3f> &points,
                             bool center);

template <int Dim>
void RotatePoints(const Eigen::Matrix<float, Dim, Dim> &R,
                  utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
                  bool center) {
    RotatePoints<Dim>(0, R, points, center);
}

template <int Dim>
void RotatePoints(cudaStream_t stream,
                  const Eigen::Matrix<float, Dim, Dim> &R,
                  utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points,
                  bool center) {
    Eigen::Matrix<float, Dim, 1> points_center =
            Eigen::Matrix<float, Dim, 1>::Zero();
    if (center && !points.empty()) {
        points_center = ComputeCenter<Dim>(points);
    }
    thrust::for_each(utility::exec_policy(stream), points.begin(),
                     points.end(),
                     [=] __device__(Eigen::Matrix<float, Dim, 1> & pt) {
                         pt = R * (pt - points_center) + points_center;
                     });
}

template void RotatePoints<2>(const Eigen::Matrix2f &R,
                              utility::device_vector<Eigen::Vector2f> &points,
                              bool center);
template void RotatePoints<3>(const Eigen::Matrix3f &R,
                              utility::device_vector<Eigen::Vector3f> &points,
                              bool center);

template void RotatePoints<2>(cudaStream_t stream,
                              const Eigen::Matrix2f &R,
                              utility::device_vector<Eigen::Vector2f> &points,
                              bool center);
template void RotatePoints<3>(cudaStream_t stream,
                              const Eigen::Matrix3f &R,
                              utility::device_vector<Eigen::Vector3f> &points,
                              bool center);

void RotateNormals(const Eigen::Matrix3f &R,
                   utility::device_vector<Eigen::Vector3f> &normals) {
    RotateNormals(0, R, normals);
}

void RotateNormals(cudaStream_t stream,
                   const Eigen::Matrix3f &R,
                   utility::device_vector<Eigen::Vector3f> &normals) {
    thrust::for_each(utility::exec_policy(stream), normals.begin(),
                     normals.end(), [=] __device__(Eigen::Vector3f & normal) {
                         normal = R * normal;
                     });
}

Eigen::Matrix3f GetRotationMatrixFromXYZ(const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixX(rotation(0)) *
           cupoch::utility::RotationMatrixY(rotation(1)) *
           cupoch::utility::RotationMatrixZ(rotation(2));
}

Eigen::Matrix3f GetRotationMatrixFromYZX(const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixY(rotation(0)) *
           cupoch::utility::RotationMatrixZ(rotation(1)) *
           cupoch::utility::RotationMatrixX(rotation(2));
}

Eigen::Matrix3f GetRotationMatrixFromZXY(const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixZ(rotation(0)) *
           cupoch::utility::RotationMatrixX(rotation(1)) *
           cupoch::utility::RotationMatrixY(rotation(2));
}

Eigen::Matrix3f GetRotationMatrixFromXZY(const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixX(rotation(0)) *
           cupoch::utility::RotationMatrixZ(rotation(1)) *
           cupoch::utility::RotationMatrixY(rotation(2));
}

Eigen::Matrix3f GetRotationMatrixFromZYX(const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixZ(rotation(0)) *
           cupoch::utility::RotationMatrixY(rotation(1)) *
           cupoch::utility::RotationMatrixX(rotation(2));
}

Eigen::Matrix3f GetRotationMatrixFromYXZ(const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixY(rotation(0)) *
           cupoch::utility::RotationMatrixX(rotation(1)) *
           cupoch::utility::RotationMatrixZ(rotation(2));
}

Eigen::Matrix3f GetRotationMatrixFromAxisAngle(
        const Eigen::Vector3f &rotation) {
    const float phi = rotation.norm();
    return Eigen::AngleAxisf(phi, rotation / phi).toRotationMatrix();
}

Eigen::Matrix3f GetRotationMatrixFromQuaternion(
        const Eigen::Vector4f &rotation) {
    return Eigen::Quaternionf(rotation(0), rotation(1), rotation(2),
                              rotation(3))
            .normalized()
            .toRotationMatrix();
}

}  // namespace geometry
}  // namespace cupoch