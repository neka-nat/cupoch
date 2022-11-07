#pragma once
#include <limits>
#include "lbvh_index/aabb.cuh"
#include "lbvh_index/morton_code.cuh"

namespace lbvh {
        struct __align__(16) BVHNode // size: 48 bytes
        {
            AABB bounds; // 24 bytes
            unsigned int parent; // 4
            unsigned int child_left; // 4 bytes
            unsigned int child_right; // 4 bytes
            int atomic; // 4
            unsigned int range_left; // 4
            unsigned int range_right; // 4
        };

        __device__
        inline HashType morton_code(const float3& point, const lbvh::AABB &extent, float resolution = 1024.0) noexcept {
            float3 p = point;

            // scale to [0, 1]
            p.x -= extent.min.x;
            p.y -= extent.min.y;
            p.z -= extent.min.z;

            p.x /= (extent.max.x - extent.min.x);
            p.y /= (extent.max.y - extent.min.y);
            p.z /= (extent.max.z - extent.min.z);
            return morton_code(p, resolution);
        }

        __device__
        inline HashType morton_code(const lbvh::AABB &box, const lbvh::AABB &extent, float resolution = 1024.0) noexcept {
            auto p = centroid(box);
            return morton_code(p, extent, resolution);
        }


        __device__ inline bool is_leaf(const BVHNode* node) {
            return node->child_right == UINT_MAX && node->child_right == UINT_MAX;
        }

        // Sets the bounding box and traverses to root
        __device__ void process_parent(unsigned int node_idx,
                                       BVHNode* nodes,
                                       const unsigned long long int *morton_codes,
                                       unsigned int* root_index,
                                       unsigned int N)
        {
            unsigned int current_idx = node_idx;
            BVHNode* current_node = &nodes[current_idx];

            while(true) {
                // Allow only one thread to process a node
                if (atomicAdd(&(current_node->atomic), 1) != 1)
                    //printf("Terminating at node %u\n", node_idx);
                    return; // terminate the first thread encountering this

                //printf("Processing node %u\n", current_idx);
                //printf("Node %u, is leaf: %u\n", current_idx, is_leaf(current_node));
                //printf("Node %u children: %u, %u\n", current_idx, current_node->child_left, current_node->child_right);

                unsigned int left = current_node->range_left;
                unsigned int right = current_node->range_right;
                //printf("Range of node %u: %u, %u\n", current_idx, left, right);

                // Set bounding box if the node is no leaf
                if (!is_leaf(current_node)) {
                    // Fuse bounding box from children AABBs
                    current_node->bounds = merge(nodes[current_node->child_left].bounds,
                                                 nodes[current_node->child_right].bounds);
                }


                if (left == 0 && right == N - 1) {
                    root_index[0] = current_idx; // return the root
                    return; // at the root, abort
                }


                unsigned int parent_idx;
                BVHNode *parent;

                if (left == 0 || (right != N - 1 && highest_bit(morton_codes[right], morton_codes[right + 1]) <
                                                    highest_bit(morton_codes[left - 1], morton_codes[left]))) {
                    // parent = right, set parent left child and range to node
                    parent_idx = N + right;

                    parent = &nodes[parent_idx];
                    parent->child_left = current_idx;
                    parent->range_left = left;
                } else {
                    // parent = left -1, set parent right child and range to node
                    parent_idx = N + left - 1;

                    parent = &nodes[parent_idx];
                    parent->child_right = current_idx;
                    parent->range_right = right;
                }

                current_node->parent = parent_idx; // store the parent in the current node

                // up to the parent next
                current_node = parent;
                current_idx = parent_idx;
            }
        }

    /**
     * Merge an internal node into a leaf node using the leftmost leaf node of the subtree
     * @tparam T
     * @param node
     * @param leaf
     */
    __forceinline__ __device__ void make_leaf(unsigned int node_idx,
                                              unsigned int leaf_idx,
                                              BVHNode* nodes, unsigned int N) {
        BVHNode* node = &nodes[node_idx];

        unsigned int parent_idx = node->parent;

        BVHNode* leaf = &nodes[leaf_idx];

        leaf->parent = parent_idx;
        leaf->bounds = node->bounds;

        // copy the range into the leaf
        leaf->range_left = node->range_left;
        leaf->range_right = node->range_right;

        // adjust the structure at the node's parent
        if(parent_idx != UINT_MAX) {
            BVHNode* parent = &nodes[parent_idx];

            if(parent->child_left == node_idx) {
                parent->child_left = leaf_idx;
            } else {
                parent->child_right = leaf_idx;
            }
        }
    }
};