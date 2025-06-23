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
#include <Eigen/Core>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include "cupoch/knn/lbvh_knn.h"

#include <lbvh_index/aabb.cuh>
#include <lbvh_index/lbvh.cuh>
#include <lbvh_index/lbvh_kernels.cuh>
#include <lbvh_index/query_knn_kernels.cuh>

#include "cupoch/knn/kdtree_search_param.h"
#include "cupoch/utility/eigen.h"
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

template<typename T>
struct convert_float3_and_aabb_functor {
    convert_float3_and_aabb_functor() {}
    __device__
    thrust::tuple<lbvh::AABB, float3> operator() (const T& x) {
        float3 xf3 = make_float3(x[0], x[1], x[2]);
        lbvh::AABB aabb;
        aabb.min = make_float3(x[0], x[1], x[2]);
        aabb.max = make_float3(x[0], x[1], x[2]);
        return thrust::make_tuple(aabb, xf3);
    }
};

}

namespace cupoch {
namespace knn {

LinearBoundingVolumeHierarchyKNN::LinearBoundingVolumeHierarchyKNN(size_t leaf_size, bool compact, bool sort_queries, bool shrink_to_fit)
    : leaf_size_(leaf_size), compact_(compact), sort_queries_(sort_queries), shrink_to_fit_(shrink_to_fit) {
    nodes_ = std::make_unique<utility::device_vector<lbvh::BVHNode>>();
}

LinearBoundingVolumeHierarchyKNN::LinearBoundingVolumeHierarchyKNN(const utility::device_vector<Eigen::Vector3f> &data,
                                                                   size_t leaf_size, bool compact, bool sort_queries, bool shrink_to_fit)
    : leaf_size_(leaf_size), compact_(compact), sort_queries_(sort_queries), shrink_to_fit_(shrink_to_fit) {
    nodes_ = std::make_unique<utility::device_vector<lbvh::BVHNode>>();
    SetRawData(data);
}

LinearBoundingVolumeHierarchyKNN::LinearBoundingVolumeHierarchyKNN(const std::vector<Eigen::Vector3f> &data,
                                                                   size_t leaf_size, bool compact, bool sort_queries, bool shrink_to_fit)
    : leaf_size_(leaf_size), compact_(compact), sort_queries_(sort_queries), shrink_to_fit_(shrink_to_fit) {
    nodes_ = std::make_unique<utility::device_vector<lbvh::BVHNode>>();
    SetRawData(utility::device_vector<Eigen::Vector3f>(data));
}

LinearBoundingVolumeHierarchyKNN::~LinearBoundingVolumeHierarchyKNN() {}

template <typename T>
int LinearBoundingVolumeHierarchyKNN::SearchNN(const utility::device_vector<T> &query,
                                               float radius,
                                               utility::device_vector<unsigned int> &indices,
                                               utility::device_vector<float> &distance2) const{
    if (query.empty() || n_points_ <= 0 || n_nodes_ <= 0)
        return -1;
    T query0 = query[0];
    if (size_t(query0.size()) != dimension_) return -1;
    return SearchNN<typename utility::device_vector<T>::const_iterator,
                    T::RowsAtCompileTime>(query.begin(), query.end(), radius,
                                          indices, distance2);
}

template <typename T>
int LinearBoundingVolumeHierarchyKNN::SearchNN(const T &query,
                           float radius,
                           thrust::host_vector<unsigned int> &indices,
                           thrust::host_vector<float> &distance2) const {
    utility::device_vector<T> query_dv(1, query);
    utility::device_vector<unsigned int> indices_dv;
    utility::device_vector<float> distance2_dv;
    auto result = SearchNN<T>(query_dv, radius, indices_dv, distance2_dv);
    indices = indices_dv;
    distance2 = distance2_dv;
    return result;
}

template <typename T>
int LinearBoundingVolumeHierarchyKNN::SearchNN(const T &query,
                           float radius,
                           std::vector<unsigned int> &indices,
                           std::vector<float> &distance2) const {
    utility::device_vector<T> query_dv(1, query);
    utility::device_vector<unsigned int> indices_dv;
    utility::device_vector<float> distance2_dv;
    auto result = SearchNN<T>(query_dv, radius, indices_dv, distance2_dv);
    indices.resize(indices_dv.size());
    distance2.resize(distance2_dv.size());
    copy_device_to_host(indices_dv, indices);
    copy_device_to_host(distance2_dv, distance2);
    return result;
}

template <typename T>
bool LinearBoundingVolumeHierarchyKNN::SetRawData(const utility::device_vector<T> &data) {
    n_points_ = data.size();
    n_nodes_ = n_points_ * 2 - 1;
    dimension_ = T::SizeAtCompileTime;
    data_float3_.resize(n_points_);
    dim3 block_dim, grid_dim;
    std::tie(block_dim, grid_dim) = utility::SelectBlockGridSizes(n_points_);

    // init aabbs
    utility::device_vector<lbvh::AABB> aabbs(n_points_);
    thrust::transform(data.begin(), data.end(), make_tuple_begin(aabbs, data_float3_), convert_float3_and_aabb_functor<T>());
    T min_data = utility::ComputeMinBound<T::SizeAtCompileTime, typename T::Scalar>(data);
    T max_data = utility::ComputeMaxBound<T::SizeAtCompileTime, typename T::Scalar>(data);
    extent_.first = Eigen::Vector3f(min_data[0], min_data[1], min_data[2]);
    extent_.second = Eigen::Vector3f(max_data[0], max_data[1], max_data[2]);
    auto extent_float3 = to_float3_aabb(extent_);

    // compute the morton codes of the aabbs
    utility::device_vector<lbvh::HashType> morton_codes(n_points_);
    thrust::transform(
        aabbs.begin(), aabbs.end(), morton_codes.begin(),
        [extent_float3] __device__ (const lbvh::AABB& aabb) { return lbvh::morton_code(aabb, extent_float3); });

    // sort everything by the morton codes
    sorted_indices_.resize(morton_codes.size());
    thrust::sequence(sorted_indices_.begin(), sorted_indices_.end());
    thrust::sort_by_key(morton_codes.begin(), morton_codes.end(), make_tuple_begin(sorted_indices_, aabbs));

    // allocate space for the nodes as a raw cuda array
    nodes_->resize(n_nodes_);
    initialize_tree_kernel<<<grid_dim, block_dim>>>(
        thrust::raw_pointer_cast(nodes_->data()), thrust::raw_pointer_cast(aabbs.data()), n_points_);
    cudaSafeCall(cudaDeviceSynchronize());
    thrust::device_vector<unsigned int> root_node_index(1, std::numeric_limits<unsigned int>::max());
    construct_tree_kernel<<<grid_dim, block_dim>>>(
        thrust::raw_pointer_cast(nodes_->data()),
        thrust::raw_pointer_cast(root_node_index.data()),
        thrust::raw_pointer_cast(morton_codes.data()), n_points_);
    cudaSafeCall(cudaDeviceSynchronize());

    if (leaf_size_ > 1) {
        utility::device_vector<unsigned int> valid(n_nodes_, 1);
        optimize_tree_kernel<<<grid_dim, block_dim>>>(
            thrust::raw_pointer_cast(nodes_->data()),
            thrust::raw_pointer_cast(root_node_index.data()),
            thrust::raw_pointer_cast(valid.data()), leaf_size_, n_points_);
        cudaSafeCall(cudaDeviceSynchronize());
        // compact the tree to increase bandwidth
        if (compact_) {
            utility::device_vector<unsigned int> valid_sums(n_nodes_ + 1, 0);
            thrust::inclusive_scan(valid.begin(), valid.end(), valid_sums.begin() + 1);
            int new_node_count = valid_sums[n_nodes_];
            utility::device_vector<unsigned int> valid_sums_aligned(valid_sums.begin(), valid_sums.end() - 1);
            utility::device_vector<unsigned int> isum(n_nodes_);
            thrust::transform(
                enumerate_begin(valid_sums_aligned), enumerate_end(valid_sums_aligned), isum.begin(),
                [] __device__ (const thrust::tuple<unsigned int, unsigned int>& x) { return thrust::get<0>(x) - thrust::get<1>(x); });
            unsigned int free_indices_size = isum[new_node_count];
            utility::device_vector<unsigned int> free(valid_sums);
            free.resize(new_node_count);
            std::tie(block_dim, grid_dim) = utility::SelectBlockGridSizes(new_node_count);
            compute_free_indices_kernel<<<grid_dim, block_dim>>>(
                thrust::raw_pointer_cast(valid_sums.data()), thrust::raw_pointer_cast(isum.data()), thrust::raw_pointer_cast(free.data()), new_node_count);

            unsigned int first_moved = valid_sums[new_node_count];
            std::tie(block_dim, grid_dim) = utility::SelectBlockGridSizes(n_nodes_);
            compact_tree_kernel<<<grid_dim, block_dim>>>(
                thrust::raw_pointer_cast(nodes_->data()),
                thrust::raw_pointer_cast(root_node_index.data()),
                thrust::raw_pointer_cast(valid_sums.data()),
                thrust::raw_pointer_cast(free.data()),
                first_moved, new_node_count, n_nodes_);
            if (shrink_to_fit_) {
                nodes_->resize(new_node_count);
            }
            n_nodes_ = new_node_count;
        }
        root_node_index_ = root_node_index[0];
    }
    return true;
}

template <typename InputIterator, int Dim>
int LinearBoundingVolumeHierarchyKNN::SearchNN(InputIterator first,
                                               InputIterator last,
                                               float radius,
                                               utility::device_vector<unsigned int> &indices,
                                               utility::device_vector<float> &distance2) const {
    size_t num_query = thrust::distance(first, last);
    auto extent_float3 = to_float3_aabb(extent_);
    utility::device_vector<float3> data_float3(num_query);
    thrust::transform(first, last, data_float3.begin(), convert_float3_functor<Dim>());

    utility::device_vector<lbvh::HashType> morton_codes(num_query);
    utility::device_vector<unsigned int> sorted_indices(num_query);
    thrust::sequence(sorted_indices.begin(), sorted_indices.end());
    if (sort_queries_) {
        dim3 block_dim, grid_dim;
        std::tie(block_dim, grid_dim) = utility::SelectBlockGridSizes(num_query);
        compute_morton_points_kernel<<<grid_dim, block_dim>>>(
            thrust::raw_pointer_cast(data_float3.data()), extent_float3, thrust::raw_pointer_cast(morton_codes.data()), num_query);
        cudaSafeCall(cudaDeviceSynchronize());
        thrust::sort_by_key(morton_codes.begin(), morton_codes.end(), sorted_indices.begin());
    }

    dim3 block_dim, grid_dim;
    std::tie(block_dim, grid_dim) = utility::SelectBlockGridSizes(num_query);
    indices.resize(num_query, std::numeric_limits<unsigned int>::max());
    distance2.resize(num_query, std::numeric_limits<float>::max());
    utility::device_vector<unsigned int> neighbors(num_query, 0);

    query_knn_kernel<<<grid_dim, block_dim>>>(
        thrust::raw_pointer_cast(nodes_->data()),
        thrust::raw_pointer_cast(data_float3_.data()),
        thrust::raw_pointer_cast(sorted_indices_.data()),
        root_node_index_,
        radius * radius,
        thrust::raw_pointer_cast(data_float3.data()),
        thrust::raw_pointer_cast(sorted_indices.data()),
        num_query,
        thrust::raw_pointer_cast(indices.data()),
        thrust::raw_pointer_cast(distance2.data()),
        thrust::raw_pointer_cast(neighbors.data()));
    cudaSafeCall(cudaDeviceSynchronize());
    return 1;
}

template int LinearBoundingVolumeHierarchyKNN::SearchNN<Eigen::Vector3f>(
        const utility::device_vector<Eigen::Vector3f> &query,
        float radius,
        utility::device_vector<unsigned int> &indices,
        utility::device_vector<float> &distance2) const;

template int LinearBoundingVolumeHierarchyKNN::SearchNN<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        float radius,
        thrust::host_vector<unsigned int> &indices,
        thrust::host_vector<float> &distance2) const;

template int LinearBoundingVolumeHierarchyKNN::SearchNN<Eigen::Vector3f>(
        const Eigen::Vector3f &query,
        float radius,
        std::vector<unsigned int> &indices,
        std::vector<float> &distance2) const;

template bool LinearBoundingVolumeHierarchyKNN::SetRawData<Eigen::Vector3f>(
        const utility::device_vector<Eigen::Vector3f> &data);

}
}