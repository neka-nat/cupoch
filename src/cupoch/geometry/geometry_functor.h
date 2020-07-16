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

#include <Eigen/Core>

namespace cupoch {
namespace geometry {

struct compute_grid_center_functor {
    compute_grid_center_functor(float voxel_size, const Eigen::Vector3f& origin)
        : voxel_size_(voxel_size),
          origin_(origin),
          half_voxel_size_(
                  0.5 * voxel_size, 0.5 * voxel_size, 0.5 * voxel_size){};
    const float voxel_size_;
    const Eigen::Vector3f origin_;
    const Eigen::Vector3f half_voxel_size_;
    __device__ Eigen::Vector3f operator()(const Eigen::Vector3i& x) const {
        return x.cast<float>() * voxel_size_ + origin_ + half_voxel_size_;
    }
};

template <typename TupleType, int Index, typename Func>
struct tuple_element_compare_functor {
    __device__ bool operator()(const TupleType& rhs, const TupleType& lhs) {
        return Func()(thrust::get<Index>(rhs), thrust::get<Index>(lhs));
    }
};

}  // namespace geometry
}  // namespace cupoch