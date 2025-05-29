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
    copy_device_to_host(trianglemesh.vertices_, vertices_);
    copy_device_to_host(trianglemesh.vertex_normals_, vertex_normals_);
    copy_device_to_host(trianglemesh.vertex_colors_, vertex_colors_);
    copy_device_to_host(trianglemesh.triangles_, triangles_);
    copy_device_to_host(trianglemesh.triangle_normals_, triangle_normals_);
    copy_device_to_host(trianglemesh.triangle_uvs_, triangle_uvs_);
    texture_.FromDevice(trianglemesh.texture_);
}

void HostTriangleMesh::ToDevice(geometry::TriangleMesh& trianglemesh) const {
    trianglemesh.vertices_.resize(vertices_.size());
    trianglemesh.vertex_normals_.resize(vertex_normals_.size());
    trianglemesh.vertex_colors_.resize(vertex_colors_.size());
    trianglemesh.triangles_.resize(triangles_.size());
    trianglemesh.triangle_normals_.resize(triangle_normals_.size());
    trianglemesh.triangle_uvs_.resize(triangle_uvs_.size());
    copy_host_to_device(vertices_, trianglemesh.vertices_);
    copy_host_to_device(vertex_normals_, trianglemesh.vertex_normals_);
    copy_host_to_device(vertex_colors_, trianglemesh.vertex_colors_);
    copy_host_to_device(triangles_, trianglemesh.triangles_);
    copy_host_to_device(triangle_normals_, trianglemesh.triangle_normals_);
    copy_host_to_device(triangle_uvs_, trianglemesh.triangle_uvs_);
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