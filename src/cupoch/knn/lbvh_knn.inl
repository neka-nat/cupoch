/**
 * Copyright (c) 2022 Neka-Nat
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
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include "lbvh_knn.h"
#include <lbvh_index/aabb.cuh>
#include <lbvh_index/lbvh.cuh>
#include <lbvh_index/lbvh_kernels.cuh>
#include <lbvh_index/query_knn_kernels.cuh>

#include "cupoch/utility/platform.h"
#include "cupoch/utility/helper.h"

namespace {

template<int Dim>
struct convert_float3_functor {
    convert_float3_functor() {}
    __device__
    float3 operator() (const Eigen::Matrix<float, Dim, 1>& x) {
        return make_float3(x[0], x[1], x[2]);
    }
};

__device__ __host__
lbvh::AABB to_float3_aabb(const cupoch::knn::AABB& aabb) {
    lbvh::AABB aabb_f3;
    aabb_f3.min = make_float3(aabb.first[0], aabb.first[1], aabb.first[2]);
    aabb_f3.max = make_float3(aabb.second[0], aabb.second[1], aabb.second[2]);
    return aabb_f3;
}

}
namespace cupoch {
namespace knn {

template <typename InputIterator, int Dim>
int LinearBoundingVolumeHierarchyKNN::SearchKNN(InputIterator first,
                                                InputIterator last,
                                                int knn,
                                                utility::device_vector<unsigned int> &indices,
                                                utility::device_vector<float> &distance2) const {
    size_t num_query = thrust::distance(first, last);
    utility::device_vector<float3> data_float3(num_query);
    thrust::transform(first, last, data_float3.begin(), convert_float3_functor<Dim>());

    utility::device_vector<unsigned long long int> morton_codes(num_query);
    utility::device_vector<unsigned int> sorted_indices(num_query);
    thrust::sequence(sorted_indices.begin(), sorted_indices.end());
    if (sort_queries_) {
        dim3 block_dim, grid_dim;
        std::tie(block_dim, grid_dim) = utility::SelectBlockGridSizes(num_query);
        compute_morton_points_kernel<<<grid_dim, block_dim>>>(
            thrust::raw_pointer_cast(data_float3.data()), to_float3_aabb(extent_), thrust::raw_pointer_cast(morton_codes.data()), num_query);
        cudaSafeCall(cudaDeviceSynchronize());
        thrust::sort_by_key(morton_codes.begin(), morton_codes.end(), sorted_indices.begin());
    }

    dim3 block_dim, grid_dim;
    std::tie(block_dim, grid_dim) = utility::SelectBlockGridSizes(num_query);
    indices.resize(num_query * knn, std::numeric_limits<unsigned int>::max());
    distance2.resize(num_query * knn, std::numeric_limits<float>::max());
    utility::device_vector<unsigned int> neighbors(num_query, 0);

    query_knn_kernel<<<grid_dim, block_dim>>>(
        thrust::raw_pointer_cast(nodes_->data()),
        thrust::raw_pointer_cast(data_float3_.data()),
        thrust::raw_pointer_cast(sorted_indices_.data()),
        root_node_index_,
        std::numeric_limits<float>::max(),
        thrust::raw_pointer_cast(data_float3.data()),
        thrust::raw_pointer_cast(sorted_indices.data()),
        num_query,
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(distance2.data()),
        thrust::raw_pointer_cast(neighbors.data()));
    cudaSafeCall(cudaDeviceSynchronize());
    return 1;
}

}
}