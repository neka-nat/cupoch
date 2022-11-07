#pragma once
#define HASH_64 1 // use 64 bit morton codes

#include "lbvh_index/lbvh.cuh"


__global__ void compute_morton_points_kernel(float3* __restrict__ const points,
                                             lbvh::AABB extent,
                                             unsigned long long int* morton_codes,
                                             unsigned int N) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N)
        return;

    const float3& point = points[idx];
    morton_codes[idx] = lbvh::morton_code(point, extent);
}

__forceinline__ __device__ void initialize_leaf_node(unsigned int leaf_idx, lbvh::BVHNode *nodes, const lbvh::AABB *sorted_aabbs) {
    // Reset leaf nodes
    lbvh::BVHNode* leaf = &nodes[leaf_idx];

    leaf->bounds = sorted_aabbs[leaf_idx];
    leaf->atomic = 1; // leaf nodes will be processed by the first thread
    leaf->range_right = leaf_idx;
    leaf->range_left = leaf_idx;
    leaf->parent = UINT_MAX;
    leaf->child_left = UINT_MAX;
    leaf->child_right = UINT_MAX;
}

__forceinline__ __device__ void initialize_internal_node(unsigned int internal_index, lbvh::BVHNode *nodes) {


    auto* internal = &nodes[internal_index];
    internal->atomic = 0; // internal nodes will be processed by the second thread encountering them
    internal->parent = UINT_MAX;
    internal->child_left = UINT_MAX;
    internal->child_right = UINT_MAX;
}

__global__ void initialize_tree_kernel(lbvh::BVHNode *nodes,
                                       const lbvh::AABB *sorted_aabbs,
                                       unsigned int N)
{
    unsigned int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= N)
        return;

    // Reset leaf nodes
    initialize_leaf_node(leaf_idx, nodes, sorted_aabbs);

    // Reset internal nodes
    if(leaf_idx < N-1) {
        // Reset internal nodes
        unsigned int internal_index = N + leaf_idx;
        initialize_internal_node(internal_index, nodes);
    }
}

__global__ void construct_tree_kernel(lbvh::BVHNode *nodes,
                                      unsigned int* root_index,
                                      const unsigned long long int *sorted_morton_codes,
                                      unsigned int N)
{
    unsigned int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= N)
        return;

    // Special case
    if (N == 1)
    {
        lbvh::BVHNode* leaf = &nodes[leaf_idx];
        nodes[N].bounds = leaf->bounds;
        nodes[N].child_left = leaf_idx;
        root_index[0] = N;
    } else {

        // recurse up to the root building up the tree
        process_parent(leaf_idx, nodes, sorted_morton_codes, root_index, N);
    }
}

__global__ void optimize_tree_kernel(lbvh::BVHNode *nodes,
                                     unsigned int* root_index,
                                     unsigned int* valid,
                                     unsigned int max_node_size,
                                     unsigned int N)
{
    unsigned int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= N)
        return;

    lbvh::BVHNode* leaf = &nodes[leaf_idx];

    unsigned int current_idx = leaf->parent;
    lbvh::BVHNode* current = &nodes[current_idx];

    lbvh::BVHNode* parent;
    unsigned int node_size;

    while(true) {
        if(current_idx == UINT_MAX) {
            //arrived at root by merging all nodes.
            root_index[0] = leaf_idx;
            return; // we are at the root
        }
        const unsigned int parent_idx = current->parent;
        parent = &nodes[parent_idx]; // this might change due to merges

        node_size = current->range_right - current->range_left + 1;
        if(node_size <= max_node_size && leaf_idx <= (current_idx-N)) {
            // only one thread will do this
            make_leaf(current_idx, leaf_idx, nodes, N);
            current->atomic = -1; // mark the current node as invalid to make it removable by the optimization
            valid[current_idx] = 0;

            current = parent;
            current_idx = parent_idx;
        } else if(node_size <= max_node_size && leaf_idx > (current_idx-N)) {
            // the other thread will just set it's leaf to invalid
            // as it will be merged by the other thread and abort
            leaf->atomic = -1;
            valid[leaf_idx] = 0;

            return;
        } else {
            return; // nothing to do here so abort
        }
    }
}

__forceinline__ __device__ bool valid_node(const unsigned int* valid_sums, unsigned int idx)
{
    return valid_sums[idx] != valid_sums[idx+1];
}

__global__ void compute_free_indices_kernel(const unsigned int* valid_sums,
                                     const unsigned int* isums,
                                     unsigned int* free_indices,
                                     unsigned int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    if(!valid_node(valid_sums, idx)) {
        auto free_index = isums[idx];
        free_indices[free_index] = idx;
    }
}

__global__ void compact_tree_kernel(lbvh::BVHNode *nodes,
                                     unsigned int* root_index,
                                     const unsigned int* valid_sums,
                                     const unsigned int* free_positions,
                                     unsigned int first_moved,
                                     unsigned int node_cnt_new,
                                     unsigned int N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // correction of id's in non-moved part
    if (idx < node_cnt_new) {
        // if the node is valid
        if (valid_node(valid_sums, idx))
        {
            lbvh::BVHNode* node = &nodes[idx];

            unsigned int parent_idx = node->parent;
            // adjust the parent index in case the parent will be moved
            if ((parent_idx >= node_cnt_new) && (parent_idx < N)) {
                node->parent = free_positions[valid_sums[parent_idx] - first_moved];
            }

            // same with the left and right children
            unsigned int left_idx = nodes[idx].child_left;
            if((left_idx >= node_cnt_new) && (left_idx < N)) {
                node->child_left = free_positions[valid_sums[left_idx] - first_moved];
            }

            unsigned int right_idx = nodes[idx].child_right;
            if((right_idx >= node_cnt_new) && (right_idx < N)) {
                node->child_right = free_positions[valid_sums[right_idx] - first_moved];
            }
        }

    // actually move nodes into the non-moved section here
    } else if(idx >= node_cnt_new) {
        // if the node is valid
        if (valid_node(valid_sums, idx))
        {

            lbvh::BVHNode* node = &nodes[idx];
            unsigned int new_position = free_positions[valid_sums[idx] - first_moved];
            lbvh::BVHNode* new_node = &nodes[new_position];
            // copy the static parameters into the new node

            new_node->bounds = node->bounds;
            new_node->atomic = node->atomic;
            node->atomic = -1; // mark node as invalid for debugging
            new_node->range_left = node->range_left;
            new_node->range_right = node->range_right;

            // adjust the parent and child indices if required
            unsigned int parent_idx = node->parent;
            if ((parent_idx >= node_cnt_new) && (parent_idx < N)) {
                parent_idx = free_positions[valid_sums[parent_idx] - first_moved];
            }

            if(parent_idx == UINT_MAX) {
                // this node is the root node so adjust the root index to the new position
                root_index[0] = new_position;
            }

            new_node->parent = parent_idx;

            unsigned int left_idx = node->child_left;
            if ((left_idx >= node_cnt_new) && (left_idx < N)) {
                left_idx = free_positions[valid_sums[left_idx] - first_moved];
            }

            new_node->child_left = left_idx;

            unsigned int right_idx = node->child_right;
            if ((right_idx >= node_cnt_new) && (right_idx < N)) {
                right_idx = free_positions[valid_sums[right_idx] - first_moved];
            }

            new_node->child_right = right_idx;
        }
    }



}