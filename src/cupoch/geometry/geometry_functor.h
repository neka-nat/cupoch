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
#include <thrust/tuple.h>

namespace cupoch {
namespace geometry {

// Coordinates of 8 vertices in a cuboid (assume origin (0,0,0), size 1)
__constant__ int cuboid_vertex_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1},
};

struct compute_cumulant_functor {
    compute_cumulant_functor(const Eigen::Vector3f *points) : points_(points){};
    const Eigen::Vector3f *points_;
    __device__ thrust::tuple<Eigen::Matrix<float, 9, 1>, int> operator()(
            int idx) const {
        Eigen::Matrix<float, 9, 1> cm;
        cm.setZero();
        if (idx < 0) return thrust::make_tuple(cm, 0);
        const Eigen::Vector3f point = points_[idx];
        cm(0) = point(0);
        cm(1) = point(1);
        cm(2) = point(2);
        cm(3) = point(0) * point(0);
        cm(4) = point(0) * point(1);
        cm(5) = point(0) * point(2);
        cm(6) = point(1) * point(1);
        cm(7) = point(1) * point(2);
        cm(8) = point(2) * point(2);
        return thrust::make_tuple(cm, 1);
    }
};

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

template <typename VoxelType, typename IndexType>
struct get_grid_index_functor {
    __device__ IndexType operator()(const VoxelType& v) const {
        return v.grid_index_;
    }
};

template <typename IndexType>
struct compute_voxel_vertices_functor {
    compute_voxel_vertices_functor(const Eigen::Vector3f& origin,
                                   float voxel_size)
        : origin_(origin), voxel_size_(voxel_size){};
    const Eigen::Vector3f origin_;
    const float voxel_size_;
    __device__ Eigen::Vector3f operator()(
            const thrust::tuple<size_t, IndexType>& x) const {
        int j = thrust::get<0>(x);
        const IndexType grid_index = thrust::get<1>(x);
        // 8 vertices in a voxel
        Eigen::Vector3f base_vertex =
                origin_ + grid_index.template cast<float>() * voxel_size_;
        const auto offset_v = Eigen::Vector3f(cuboid_vertex_offsets[j][0],
                                              cuboid_vertex_offsets[j][1],
                                              cuboid_vertex_offsets[j][2]);
        return base_vertex + offset_v * voxel_size_;
    }
};

}  // namespace geometry
}  // namespace cupoch