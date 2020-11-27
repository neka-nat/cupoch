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
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/io/class_io/pointcloud_io.h"
#include "cupoch/utility/helper.h"
#include "cupoch/utility/platform.h"

using namespace cupoch;
using namespace cupoch::io;

void HostPointCloud::FromDevice(const geometry::PointCloud& pointcloud) {
    points_.resize(pointcloud.points_.size());
    normals_.resize(pointcloud.normals_.size());
    colors_.resize(pointcloud.colors_.size());
    cudaSafeCall(cudaMemcpy(points_.data(), thrust::raw_pointer_cast(pointcloud.points_.data()),
                            points_.size() * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(normals_.data(), thrust::raw_pointer_cast(pointcloud.normals_.data()),
                            normals_.size() * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(colors_.data(), thrust::raw_pointer_cast(pointcloud.colors_.data()),
                            colors_.size() * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost));
}

void HostPointCloud::ToDevice(geometry::PointCloud& pointcloud) const {
    pointcloud.points_.resize(points_.size());
    pointcloud.normals_.resize(normals_.size());
    pointcloud.colors_.resize(colors_.size());
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(pointcloud.points_.data()), points_.data(),
                            points_.size() * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(pointcloud.normals_.data()), normals_.data(),
                            normals_.size() * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(pointcloud.colors_.data()), colors_.data(),
                            colors_.size() * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
}

void HostPointCloud::Clear() {
    points_.clear();
    normals_.clear();
    colors_.clear();
}
