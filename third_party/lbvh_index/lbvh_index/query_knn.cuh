#pragma once
#include "query.cuh"
#include "static_priorityqueue.cuh"

// default is one nearest neighbor
#ifndef K
#define K 1
#endif

namespace lbvh {
    __device__ void query_knn(const BVHNode* __restrict__ nodes,
                             const float3* __restrict__ points,
                             const unsigned int* __restrict__ sorted_indices,
                             unsigned int root_index,
                             const float3* __restrict__ query_point,
                             StaticPriorityQueue<float, K>& queue)
    {
        query<StaticPriorityQueue<float, K>>(nodes, points, sorted_indices, root_index, query_point, queue);
    }

    __device__ StaticPriorityQueue<float, K> query_knn(const BVHNode* __restrict__ nodes,
                             const float3* __restrict__ points,
                             const unsigned int* __restrict__ sorted_indices,
                             unsigned int root_index,
                             const float3* __restrict__ query_point,
                             const float max_radius)
    {

        StaticPriorityQueue<float, K> queue(max_radius);
        query_knn(nodes, points, sorted_indices, root_index, query_point, queue);
        return std::move(queue);
    }
}

