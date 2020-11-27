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
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/io/class_io/trianglemesh_io.h"
#include "cupoch/utility/platform.h"

using namespace cupoch;
using namespace cupoch::io;

void HostTriangleMesh::FromDevice(const geometry::TriangleMesh& trianglemesh) {
    vertices_.resize(trianglemesh.vertices_.size());
    vertex_normals_.resize(trianglemesh.vertex_normals_.size());
    vertex_colors_.resize(trianglemesh.vertex_colors_.size());
    triangles_.resize(trianglemesh.triangles_.size());
    triangle_normals_.resize(trianglemesh.triangle_normals_.size());
    triangle_uvs_.resize(trianglemesh.triangle_uvs_.size());
    cudaSafeCall(cudaMemcpy(vertices_.data(), thrust::raw_pointer_cast(trianglemesh.vertices_.data()),
                            vertices_.size() * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(vertex_normals_.data(), thrust::raw_pointer_cast(trianglemesh.vertex_normals_.data()),
                            vertex_normals_.size() * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(vertex_colors_.data(), thrust::raw_pointer_cast(trianglemesh.vertex_colors_.data()),
                            vertex_colors_.size() * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(triangles_.data(), thrust::raw_pointer_cast(trianglemesh.triangles_.data()),
                            triangles_.size() * sizeof(Eigen::Vector3i), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(triangle_normals_.data(), thrust::raw_pointer_cast(trianglemesh.triangle_normals_.data()),
                            triangle_normals_.size() * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(triangle_uvs_.data(), thrust::raw_pointer_cast(trianglemesh.triangle_uvs_.data()),
                            triangle_uvs_.size() * sizeof(Eigen::Vector2f), cudaMemcpyDeviceToHost));
    texture_.FromDevice(trianglemesh.texture_);
}

void HostTriangleMesh::ToDevice(geometry::TriangleMesh& trianglemesh) const {
    trianglemesh.vertices_.resize(vertices_.size());
    trianglemesh.vertex_normals_.resize(vertex_normals_.size());
    trianglemesh.vertex_colors_.resize(vertex_colors_.size());
    trianglemesh.triangles_.resize(triangles_.size());
    trianglemesh.triangle_normals_.resize(triangle_normals_.size());
    trianglemesh.triangle_uvs_.resize(triangle_uvs_.size());
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(trianglemesh.vertices_.data()), vertices_.data(),
                            vertices_.size() * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(trianglemesh.vertex_normals_.data()), vertex_normals_.data(),
                            vertex_normals_.size() * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(trianglemesh.vertex_colors_.data()), vertex_colors_.data(),
                            vertex_colors_.size() * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(trianglemesh.triangles_.data()), triangles_.data(),
                            triangles_.size() * sizeof(Eigen::Vector3i), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(trianglemesh.triangle_normals_.data()), triangle_normals_.data(),
                            triangle_normals_.size() * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(trianglemesh.triangle_uvs_.data()), triangle_uvs_.data(),
                            triangle_uvs_.size() * sizeof(Eigen::Vector2f), cudaMemcpyHostToDevice));
    texture_.ToDevice(trianglemesh.texture_);
}

void HostTriangleMesh::Clear() {
    vertices_.clear();
    vertex_normals_.clear();
    vertex_colors_.clear();
    triangles_.clear();
    triangle_normals_.clear();
    triangle_uvs_.clear();
    texture_.Clear();
}