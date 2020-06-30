#include <Eigen/Dense>

#include "cupoch/geometry/geometry_utils.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace geometry {

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

template<int Dim>
Eigen::Matrix<float, Dim, 1> ComputeMinBound(
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points) {
    return ComputeMinBound(0, points);
}

template<int Dim>
Eigen::Matrix<float, Dim, 1> ComputeMinBound(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points) {
    if (points.empty()) return Eigen::Matrix<float, Dim, 1>::Zero();
    Eigen::Matrix<float, Dim, 1> init = points[0];
    return thrust::reduce(utility::exec_policy(stream)->on(stream),
                          points.begin(), points.end(), init,
                          thrust::elementwise_minimum<Eigen::Matrix<float, Dim, 1>>());
}

template<int Dim>
Eigen::Matrix<float, Dim, 1> ComputeMaxBound(
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points) {
    return ComputeMaxBound(0, points);
}

template<int Dim>
Eigen::Matrix<float, Dim, 1> ComputeMaxBound(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points) {
    if (points.empty()) return Eigen::Matrix<float, Dim, 1>::Zero();
    Eigen::Matrix<float, Dim, 1> init = points[0];
    return thrust::reduce(utility::exec_policy(stream)->on(stream),
                          points.begin(), points.end(), init,
                          thrust::elementwise_maximum<Eigen::Matrix<float, Dim, 1>>());
}

template<int Dim>
Eigen::Matrix<float, Dim, 1> ComputeCenter(
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points) {
    Eigen::Matrix<float, Dim, 1> init = Eigen::Matrix<float, Dim, 1>::Zero();
    if (points.empty()) return init;
    Eigen::Matrix<float, Dim, 1> sum = thrust::reduce(points.begin(), points.end(), init,
                                           thrust::plus<Eigen::Matrix<float, Dim, 1>>());
    return sum / points.size();
}

template Eigen::Matrix<float, 3, 1> ComputeMinBound(
        const utility::device_vector<Eigen::Matrix<float, 3, 1>> &points); 
template Eigen::Matrix<float, 3, 1> ComputeMinBound(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Matrix<float, 3, 1>> &points); 
    
template Eigen::Matrix<float, 3, 1> ComputeMaxBound(
        const utility::device_vector<Eigen::Matrix<float, 3, 1>> &points); 
template Eigen::Matrix<float, 3, 1> ComputeMaxBound(
        cudaStream_t stream,
        const utility::device_vector<Eigen::Matrix<float, 3, 1>> &points); 

template Eigen::Matrix<float, 3, 1> ComputeCenter(
        const utility::device_vector<Eigen::Matrix<float, 3, 1>> &points);


void ResizeAndPaintUniformColor(
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

void TransformPoints(
        const Eigen::Matrix4f &transformation,
        utility::device_vector<Eigen::Vector3f> &points) {
    TransformPoints(0, transformation, points);
}

void TransformPoints(
        cudaStream_t stream,
        const Eigen::Matrix4f &transformation,
        utility::device_vector<Eigen::Vector3f> &points) {
    transform_points_functor func(transformation);
    thrust::for_each(utility::exec_policy(stream)->on(stream), points.begin(),
                     points.end(), func);
}

void TransformNormals(
        const Eigen::Matrix4f &transformation,
        utility::device_vector<Eigen::Vector3f> &normals) {
    TransformNormals(0, transformation, normals);
}

void TransformNormals(
        cudaStream_t stream,
        const Eigen::Matrix4f &transformation,
        utility::device_vector<Eigen::Vector3f> &normals) {
    transform_normals_functor func(transformation);
    thrust::for_each(utility::exec_policy(stream)->on(stream), normals.begin(),
                     normals.end(), func);
}

void TranslatePoints(
        const Eigen::Vector3f &translation,
        utility::device_vector<Eigen::Vector3f> &points,
        bool relative) {
    Eigen::Vector3f transform = translation;
    if (!relative) {
        transform -= ComputeCenter(points);
    }
    thrust::for_each(points.begin(), points.end(),
                     [=] __device__(Eigen::Vector3f & pt) { pt += transform; });
}

void ScalePoints(const float scale,
                 utility::device_vector<Eigen::Vector3f> &points,
                 bool center) {
    Eigen::Vector3f points_center(0, 0, 0);
    if (center && !points.empty()) {
        points_center = ComputeCenter(points);
    }
    thrust::for_each(points.begin(), points.end(),
                     [=] __device__(Eigen::Vector3f & pt) {
                         pt = (pt - points_center) * scale + points_center;
                     });
}

void RotatePoints(const Eigen::Matrix3f &R,
                  utility::device_vector<Eigen::Vector3f> &points,
                  bool center) {
    RotatePoints(0, R, points, center);
}

void RotatePoints(cudaStream_t stream,
                  const Eigen::Matrix3f &R,
                  utility::device_vector<Eigen::Vector3f> &points,
                  bool center) {
    Eigen::Vector3f points_center(0, 0, 0);
    if (center && !points.empty()) {
        points_center = ComputeCenter(points);
    }
    thrust::for_each(utility::exec_policy(stream)->on(stream), points.begin(),
                     points.end(), [=] __device__(Eigen::Vector3f & pt) {
                         pt = R * (pt - points_center) + points_center;
                     });
}

void RotateNormals(
        const Eigen::Matrix3f &R,
        utility::device_vector<Eigen::Vector3f> &normals) {
    RotateNormals(0, R, normals);
}

void RotateNormals(
        cudaStream_t stream,
        const Eigen::Matrix3f &R,
        utility::device_vector<Eigen::Vector3f> &normals) {
    thrust::for_each(utility::exec_policy(stream)->on(stream), normals.begin(),
                     normals.end(), [=] __device__(Eigen::Vector3f & normal) {
                         normal = R * normal;
                     });
}

Eigen::Matrix3f GetRotationMatrixFromXYZ(
        const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixX(rotation(0)) *
           cupoch::utility::RotationMatrixY(rotation(1)) *
           cupoch::utility::RotationMatrixZ(rotation(2));
}

Eigen::Matrix3f GetRotationMatrixFromYZX(
        const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixY(rotation(0)) *
           cupoch::utility::RotationMatrixZ(rotation(1)) *
           cupoch::utility::RotationMatrixX(rotation(2));
}

Eigen::Matrix3f GetRotationMatrixFromZXY(
        const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixZ(rotation(0)) *
           cupoch::utility::RotationMatrixX(rotation(1)) *
           cupoch::utility::RotationMatrixY(rotation(2));
}

Eigen::Matrix3f GetRotationMatrixFromXZY(
        const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixX(rotation(0)) *
           cupoch::utility::RotationMatrixZ(rotation(1)) *
           cupoch::utility::RotationMatrixY(rotation(2));
}

Eigen::Matrix3f GetRotationMatrixFromZYX(
        const Eigen::Vector3f &rotation) {
    return cupoch::utility::RotationMatrixZ(rotation(0)) *
           cupoch::utility::RotationMatrixY(rotation(1)) *
           cupoch::utility::RotationMatrixX(rotation(2));
}

Eigen::Matrix3f GetRotationMatrixFromYXZ(
        const Eigen::Vector3f &rotation) {
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

}
}