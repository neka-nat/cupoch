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
#pragma once

// Avoid warning caused by redefinition of APIENTRY macro
// defined also in glfw3.h
#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/glew.h>  // Make sure glew.h is included before gl.h
#include <GLFW/glfw3.h>

#include <Eigen/Core>
#include <string>
#include <unordered_map>

namespace cupoch {
namespace visualization {
namespace gl_helper {

const static std::unordered_map<int, GLenum> texture_format_map_ = {
        {1, GL_RED}, {3, GL_RGB}, {4, GL_RGBA}};
const static std::unordered_map<int, GLenum> texture_type_map_ = {
        {1, GL_UNSIGNED_BYTE}, {2, GL_UNSIGNED_SHORT}, {4, GL_FLOAT}};

typedef Eigen::Matrix<GLfloat, 3, 1, Eigen::ColMajor> GLVector3f;
typedef Eigen::Matrix<GLfloat, 4, 1, Eigen::ColMajor> GLVector4f;
typedef Eigen::Matrix<GLfloat, 4, 4, Eigen::ColMajor> GLMatrix4f;

GLMatrix4f LookAt(const Eigen::Vector3f &eye,
                  const Eigen::Vector3f &lookat,
                  const Eigen::Vector3f &up);

GLMatrix4f Perspective(float field_of_view_,
                       float aspect,
                       float z_near,
                       float z_far);

GLMatrix4f Ortho(float left,
                 float right,
                 float bottom,
                 float top,
                 float z_near,
                 float z_far);

Eigen::Vector3f Project(const Eigen::Vector3f &point,
                        const GLMatrix4f &mvp_matrix,
                        const int width,
                        const int height);

Eigen::Vector3f Unproject(const Eigen::Vector3f &screen_point,
                          const GLMatrix4f &mvp_matrix,
                          const int width,
                          const int height);

int ColorCodeToPickIndex(const Eigen::Vector4i &color);

}  // namespace gl_helper
}  // namespace visualization
}  // namespace cupoch