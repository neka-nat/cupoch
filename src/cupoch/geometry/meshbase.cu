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
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/meshbase.h"
#include "cupoch/utility/platform.h"
#include "cupoch/utility/eigen.h"

using namespace cupoch;
using namespace cupoch::geometry;

MeshBase::MeshBase() : GeometryBase3D(Geometry::GeometryType::MeshBase) {}
MeshBase::~MeshBase() {}
MeshBase::MeshBase(const MeshBase &other)
    : GeometryBase3D(Geometry::GeometryType::MeshBase),
      vertices_(other.vertices_),
      vertex_normals_(other.vertex_normals_),
      vertex_colors_(other.vertex_colors_) {}

MeshBase &MeshBase::operator=(const MeshBase &other) {
    vertices_ = other.vertices_;
    vertex_normals_ = other.vertex_normals_;
    vertex_colors_ = other.vertex_colors_;
    return *this;
}

std::vector<Eigen::Vector3f> MeshBase::GetVertices() const {
    std::vector<Eigen::Vector3f> vertices(vertices_.size());
    copy_device_to_host(vertices_, vertices);
    return vertices;
}

void MeshBase::SetVertices(
        const thrust::host_vector<Eigen::Vector3f> &vertices) {
    vertices_ = vertices;
}

void MeshBase::SetVertices(const std::vector<Eigen::Vector3f> &vertices) {
    vertices_.resize(vertices.size());
    copy_host_to_device(vertices, vertices_);
}

std::vector<Eigen::Vector3f> MeshBase::GetVertexNormals() const {
    std::vector<Eigen::Vector3f> vertex_normals(vertex_normals_.size());
    copy_device_to_host(vertex_normals_, vertex_normals);
    return vertex_normals;
}

void MeshBase::SetVertexNormals(
        const thrust::host_vector<Eigen::Vector3f> &vertex_normals) {
    vertex_normals_ = vertex_normals;
}

void MeshBase::SetVertexNormals(const std::vector<Eigen::Vector3f> &vertex_normals) {
    vertex_normals_.resize(vertex_normals.size());
    copy_host_to_device(vertex_normals, vertex_normals_);
}

std::vector<Eigen::Vector3f> MeshBase::GetVertexColors() const {
    std::vector<Eigen::Vector3f> vertex_colors(vertex_colors_.size());
    copy_device_to_host(vertex_colors_, vertex_colors);
    return vertex_colors;
}

void MeshBase::SetVertexColors(
        const thrust::host_vector<Eigen::Vector3f> &vertex_colors) {
    vertex_colors_ = vertex_colors;
}

void MeshBase::SetVertexColors(const std::vector<Eigen::Vector3f> &vertex_colors) {
    vertex_colors_.resize(vertex_colors.size());
    copy_host_to_device(vertex_colors, vertex_colors_);
}

MeshBase &MeshBase::Clear() {
    vertices_.clear();
    vertex_normals_.clear();
    vertex_colors_.clear();
    return *this;
}

bool MeshBase::IsEmpty() const { return !HasVertices(); }

Eigen::Vector3f MeshBase::GetMinBound() const {
    return utility::ComputeMinBound<3>(vertices_);
}

Eigen::Vector3f MeshBase::GetMaxBound() const {
    return utility::ComputeMaxBound<3>(vertices_);
}

Eigen::Vector3f MeshBase::GetCenter() const {
    return utility::ComputeCenter<3>(vertices_);
}

AxisAlignedBoundingBox<3> MeshBase::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox<3>::CreateFromPoints(vertices_);
}

MeshBase &MeshBase::Transform(const Eigen::Matrix4f &transformation) {
    TransformPoints<3>(utility::GetStream(0), transformation, vertices_);
    TransformNormals(utility::GetStream(1), transformation, vertex_normals_);
    cudaDeviceSynchronize();
    return *this;
}

MeshBase &MeshBase::Translate(const Eigen::Vector3f &translation,
                              bool relative) {
    TranslatePoints<3>(translation, vertices_, relative);
    return *this;
}

MeshBase &MeshBase::Scale(const float scale, bool center) {
    ScalePoints<3>(scale, vertices_, center);
    return *this;
}

MeshBase &MeshBase::Rotate(const Eigen::Matrix3f &R, bool center) {
    RotatePoints<3>(utility::GetStream(0), R, vertices_, center);
    RotateNormals(utility::GetStream(0), R, vertex_normals_);
    cudaDeviceSynchronize();
    return *this;
}

MeshBase &MeshBase::operator+=(const MeshBase &mesh) {
    if (mesh.IsEmpty()) return (*this);
    size_t old_vert_num = vertices_.size();
    size_t add_vert_num = mesh.vertices_.size();
    size_t new_vert_num = old_vert_num + add_vert_num;
    if ((!HasVertices() || HasVertexNormals()) && mesh.HasVertexNormals()) {
        vertex_normals_.resize(new_vert_num);
        thrust::copy(mesh.vertex_normals_.begin(), mesh.vertex_normals_.end(),
                     vertex_normals_.begin() + old_vert_num);
    } else {
        vertex_normals_.clear();
    }
    if ((!HasVertices() || HasVertexColors()) && mesh.HasVertexColors()) {
        vertex_colors_.resize(new_vert_num);
        thrust::copy(mesh.vertex_colors_.begin(), mesh.vertex_colors_.end(),
                     vertex_colors_.begin() + old_vert_num);
    } else {
        vertex_colors_.clear();
    }
    vertices_.resize(new_vert_num);
    thrust::copy(mesh.vertices_.begin(), mesh.vertices_.end(),
                 vertices_.begin() + old_vert_num);
    return (*this);
}

MeshBase MeshBase::operator+(const MeshBase &mesh) const {
    return (MeshBase(*this) += mesh);
}

MeshBase &MeshBase::NormalizeNormals() {
    thrust::for_each(vertex_normals_.begin(), vertex_normals_.end(),
                     [] __device__(Eigen::Vector3f & nl) {
                         nl.normalize();
                         if (isnan(nl(0))) {
                             nl = Eigen::Vector3f(0.0, 0.0, 1.0);
                         }
                     });
    return *this;
}

MeshBase &MeshBase::PaintUniformColor(const Eigen::Vector3f &color) {
    ResizeAndPaintUniformColor(vertex_colors_, vertices_.size(), color);
    return *this;
}

MeshBase::MeshBase(Geometry::GeometryType type) : GeometryBase3D(type) {}

MeshBase::MeshBase(Geometry::GeometryType type,
                   const utility::device_vector<Eigen::Vector3f> &vertices)
    : GeometryBase3D(type), vertices_(vertices) {}

MeshBase::MeshBase(
        Geometry::GeometryType type,
        const utility::device_vector<Eigen::Vector3f> &vertices,
        const utility::device_vector<Eigen::Vector3f> &vertex_normals,
        const utility::device_vector<Eigen::Vector3f> &vertex_colors)
    : GeometryBase3D(type),
      vertices_(vertices),
      vertex_normals_(vertex_normals),
      vertex_colors_(vertex_colors) {}

MeshBase::MeshBase(Geometry::GeometryType type,
                   const thrust::host_vector<Eigen::Vector3f> &vertices)
    : GeometryBase3D(type), vertices_(vertices) {}

MeshBase::MeshBase(Geometry::GeometryType type,
                   const std::vector<Eigen::Vector3f> &vertices)
    : GeometryBase3D(type), vertices_(vertices) {}
