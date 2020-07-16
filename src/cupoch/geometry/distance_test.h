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

#include <cuda.h>

#include <Eigen/Dense>

namespace cupoch {
namespace geometry {

namespace distance_test {
__host__ __device__ inline float PointLine(const Eigen::Vector3f &p,
                                           const Eigen::Vector3f &q1,
                                           const Eigen::Vector3f &q2);

__host__ __device__ inline float PointPlane(const Eigen::Vector3f &p,
                                            const Eigen::Vector3f &vert0,
                                            const Eigen::Vector3f &vert1,
                                            const Eigen::Vector3f &vert2);

__host__ __device__ inline float PointAABBSquared(
        const Eigen::Vector3f &p,
        const Eigen::Vector3f &min_bound,
        const Eigen::Vector3f &max_bound);

__host__ __device__ inline float LineSegmentLineSegmentSquared(
        const Eigen::Vector3f &p0,
        const Eigen::Vector3f &q0,
        const Eigen::Vector3f &p1,
        const Eigen::Vector3f &q1,
        float &param0,
        float &param1,
        Eigen::Vector3f &c0,
        Eigen::Vector3f &c1);

}  // namespace distance_test

}  // namespace geometry
}  // namespace cupoch

#include "cupoch/geometry/distance_test.inl"