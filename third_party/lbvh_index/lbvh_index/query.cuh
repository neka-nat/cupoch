#pragma once
#include "lbvh.cuh"
#include "static_priorityqueue.cuh"
#include "vec_math.h"

// default is one nearest neighbor
#ifndef K
#define K 1
#endif

namespace lbvh {

    template<typename Handler>
    __forceinline__ __device__ void handle_node(const BVHNode *node,
                                                        const float3* __restrict__ points,
                                                        const unsigned int* __restrict__ sorted_indices,
                                                        const float3* __restrict__ query_point,
                                                        Handler& handler)
    {
        for(int i=node->range_left; i<=node->range_right; ++i) { // range is inclusive!
            auto index = sorted_indices[i];
            auto point = points[index];
            float dist = sq_length3(point-*query_point);

            if(dist <= handler.max_distance()) {
                handler(point, index, dist);
            }
        }
    }

    /**
     * Query the bvh using the specified handler
     * @tparam Handler functor template with the following contract:
     *                          struct Handler {
     *                              // the current number of points in the handler (can be 0)
     *                              __device__ unsigned int size() const;
     *
     *                              // the maximum number of points the handler manages (UINT_MAX for infinite
     *                              __device__ unsigned int max_size() const;
     *
     *                              // the maximum distance the handler manages (INFINITY for unbounded)
     *                              __device__ float max_distance() const;
     *
     *                              // add another point to the handler
     *                              __device__ void operator(const float3& point, unsigned int index, float dist);
     *                          }
     */
    template<typename Handler>
    __device__ void query(const BVHNode* __restrict__ nodes,
                             const float3* __restrict__ points,
                             const unsigned int* __restrict__ sorted_indices,
                             unsigned int root_index,
                             const float3* __restrict__ query_point,
                             Handler& handler)
    {
        bool bt = false;
        unsigned int last_idx = UINT_MAX;
        unsigned int current_idx = root_index;

        const BVHNode* current = &nodes[current_idx];

        unsigned int parent_idx;

        do {
            parent_idx = current->parent;
            const auto parent = &nodes[parent_idx];

            const unsigned int child_l = current->child_left;
            const unsigned int child_r = current->child_right;

            const auto child_left = &nodes[child_l];
            const auto child_right = &nodes[child_r];
            const float dl = dist_2_aabb(*query_point, child_left->bounds);
            const float dr = dist_2_aabb(*query_point, child_right->bounds);

            if(!bt && is_leaf(child_left) && dl <= handler.max_distance()) {
                handle_node(child_left, points, sorted_indices, query_point, handler);
            }
            if(!bt && is_leaf(child_right) && dr <= handler.max_distance()) {
                handle_node(child_right, points, sorted_indices, query_point, handler);
            }

            float top = handler.max_distance();
            unsigned int hsize = handler.size();
            const unsigned int max_size = handler.max_size();

            bool traverse_l = (!is_leaf(child_left) && !(hsize == max_size && dl > top));
            bool traverse_r = (!is_leaf(child_right) && !(hsize == max_size && dr > top));

            const unsigned int best_idx = (dl <= dr) ? child_l: child_r;
            const unsigned int other_idx = (dl <= dr) ? child_r: child_l;
            if(!bt) {
                if(!traverse_l && !traverse_r) {
                    // we do not traverse, so backtrack in next iteration
                    bt = true;
                    last_idx = current_idx;
                    current_idx = parent_idx;
                    current = parent;
                } else {
                    last_idx = current_idx;
                    current_idx = (traverse_l) ? child_l : child_r;
                    if (traverse_l && traverse_r) {
                        current_idx = best_idx; // take the best one if both are true
                    }
                }
            } else {
                float mind(INFINITY);

                const auto other = &nodes[other_idx];

                if(!is_leaf(other)) {
                    mind = (dl <= dr) ? dr: dl;
                }
                if(!is_leaf(other) && (last_idx == best_idx) && mind <= top) {
                    last_idx = current_idx;
                    current_idx = other_idx;
                    bt = false;
                } else {
                    last_idx = current_idx;
                    current_idx = current->parent;
                }
            }

            // get the next node
            current = &nodes[current_idx];
        } while(current_idx != UINT_MAX);

        __syncwarp(); // synchronize the warp before any other operation
    }
}

