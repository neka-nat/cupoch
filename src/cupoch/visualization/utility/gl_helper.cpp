#include "cupoch/visualization/utility/gl_helper.h"

#include <Eigen/Dense>
#include <cmath>

namespace cupoch {
namespace visualization {
namespace gl_helper {

GLMatrix4f LookAt(const Eigen::Vector3f &eye,
                  const Eigen::Vector3f &lookat,
                  const Eigen::Vector3f &up) {
    Eigen::Vector3f front_dir = (eye - lookat).normalized();
    Eigen::Vector3f up_dir = up.normalized();
    Eigen::Vector3f right_dir = up_dir.cross(front_dir).normalized();
    up_dir = front_dir.cross(right_dir).normalized();

    Eigen::Matrix4f mat = Eigen::Matrix4f::Zero();
    mat.block<1, 3>(0, 0) = right_dir.transpose();
    mat.block<1, 3>(1, 0) = up_dir.transpose();
    mat.block<1, 3>(2, 0) = front_dir.transpose();
    mat(0, 3) = -right_dir.dot(eye);
    mat(1, 3) = -up_dir.dot(eye);
    mat(2, 3) = -front_dir.dot(eye);
    mat(3, 3) = 1.0;
    return mat.cast<GLfloat>();
}

GLMatrix4f Perspective(float field_of_view_,
                       float aspect,
                       float z_near,
                       float z_far) {
    Eigen::Matrix4f mat = Eigen::Matrix4f::Zero();
    float fov_rad = field_of_view_ / 180.0 * M_PI;
    float tan_half_fov = std::tan(fov_rad / 2.0);
    mat(0, 0) = 1.0 / aspect / tan_half_fov;
    mat(1, 1) = 1.0 / tan_half_fov;
    mat(2, 2) = -(z_far + z_near) / (z_far - z_near);
    mat(3, 2) = -1.0;
    mat(2, 3) = -2.0 * z_far * z_near / (z_far - z_near);
    return mat.cast<GLfloat>();
}

GLMatrix4f Ortho(float left,
                 float right,
                 float bottom,
                 float top,
                 float z_near,
                 float z_far) {
    Eigen::Matrix4f mat = Eigen::Matrix4f::Zero();
    mat(0, 0) = 2.0 / (right - left);
    mat(1, 1) = 2.0 / (top - bottom);
    mat(2, 2) = -2.0 / (z_far - z_near);
    mat(0, 3) = -(right + left) / (right - left);
    mat(1, 3) = -(top + bottom) / (top - bottom);
    mat(2, 3) = -(z_far + z_near) / (z_far - z_near);
    mat(3, 3) = 1.0;
    return mat.cast<GLfloat>();
}

Eigen::Vector3f Project(const Eigen::Vector3f &point,
                        const GLMatrix4f &mvp_matrix,
                        const int width,
                        const int height) {
    Eigen::Vector4f pos = mvp_matrix.cast<float>() *
                          Eigen::Vector4f(point(0), point(1), point(2), 1.0);
    if (pos(3) == 0.0) {
        return Eigen::Vector3f::Zero();
    }
    pos /= pos(3);
    return Eigen::Vector3f((pos(0) * 0.5 + 0.5) * (float)width,
                           (pos(1) * 0.5 + 0.5) * (float)height,
                           (1.0 + pos(2)) * 0.5);
}

Eigen::Vector3f Unproject(const Eigen::Vector3f &screen_point,
                          const GLMatrix4f &mvp_matrix,
                          const int width,
                          const int height) {
    Eigen::Vector4f point =
            mvp_matrix.cast<float>().inverse() *
            Eigen::Vector4f(screen_point(0) / (float)width * 2.0 - 1.0,
                            screen_point(1) / (float)height * 2.0 - 1.0,
                            screen_point(2) * 2.0 - 1.0, 1.0);
    if (point(3) == 0.0) {
        return Eigen::Vector3f::Zero();
    }
    point /= point(3);
    return point.block<3, 1>(0, 0);
}

int ColorCodeToPickIndex(const Eigen::Vector4i &color) {
    if (color(0) == 255) {
        return -1;
    } else {
        return ((color(0) * 256 + color(1)) * 256 + color(2)) * 256 + color(3);
    }
}

}  // namespace gl_helper
}  // namespace visualization
}  // namespace cupoch