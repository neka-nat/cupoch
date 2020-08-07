#ifndef LBVH_QUERY_CUH
#define LBVH_QUERY_CUH
#include "predicator.cuh"

namespace lbvh
{
// query object indices that potentially overlaps with query aabb.
//
// requirements:
// - OutputIterator should be writable and its object_type should be uint32_t
//
template<typename Real, typename Objects, bool IsConst, typename OutputIterator>
__device__
unsigned int query_device(
        const detail::basic_device_bvh<Real, Objects, IsConst>& bvh,
        const query_overlap<Real> q, OutputIterator outiter,
        const unsigned int max_buffer_size = 0xFFFFFFFF) noexcept
{
    using bvh_type   = detail::basic_device_bvh<Real, Objects, IsConst>;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;

    index_type  stack[64]; // is it okay?
    index_type* stack_ptr = stack;
    *stack_ptr++ = 0; // root node is always 0
    unsigned int num_found = 0;
    const index_type root_obj_idx = bvh.nodes[0].object_idx;
    if (root_obj_idx != 0xFFFFFFFF)
    {
        if(intersects(q.target, bvh.aabbs[root_obj_idx]))
        {
            *outiter++ = root_obj_idx;
            ++num_found;
        }
        return num_found;
    }
    do
    {
        const index_type node  = *--stack_ptr;
        const index_type L_idx = bvh.nodes[node].left_idx;
        const index_type R_idx = bvh.nodes[node].right_idx;

        if(intersects(q.target, bvh.aabbs[L_idx]))
        {
            const auto obj_idx = bvh.nodes[L_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(num_found < max_buffer_size)
                {
                    *outiter++ = obj_idx;
                }
                ++num_found;
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(intersects(q.target, bvh.aabbs[R_idx]))
        {
            const auto obj_idx = bvh.nodes[R_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(num_found < max_buffer_size)
                {
                    *outiter++ = obj_idx;
                }
                ++num_found;
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    }
    while (stack < stack_ptr);
    return num_found;
}

// query object index that is the nearst to the query point.
//
// requirements:
// - DistanceCalculator must be able to calc distance between a point to an object.
//
template<typename Real, typename Objects, bool IsConst,
         typename DistanceCalculator>
__device__
thrust::pair<unsigned int, Real> query_device(
        const detail::basic_device_bvh<Real, Objects, IsConst>& bvh,
        const query_nearest<Real>& q, DistanceCalculator calc_dist) noexcept
{
    using bvh_type   = detail::basic_device_bvh<Real, Objects, IsConst>;
    using real_type  = typename bvh_type::real_type;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;

    // pair of {node_idx, mindist}
    thrust::pair<index_type, real_type>  stack[64];
    thrust::pair<index_type, real_type>* stack_ptr = stack;
    *stack_ptr++ = thrust::make_pair(0, mindist(bvh.aabbs[0], q.target));
    const auto root_obj_idx = bvh.nodes[0].object_idx;
    if (root_obj_idx != 0xFFFFFFFF)
    {
        const real_type dist = calc_dist(q.target, bvh.objects[root_obj_idx]);
        return thrust::make_pair(root_obj_idx, dist);
    }
    unsigned int nearest = 0xFFFFFFFF;
    real_type dist_to_nearest_object = infinity<real_type>();
    do
    {
        const auto node  = *--stack_ptr;
        if(node.second > dist_to_nearest_object)
        {
            // if aabb mindist > already_found_mindist, it cannot have a nearest
            continue;
        }

        const index_type L_idx = bvh.nodes[node.first].left_idx;
        const index_type R_idx = bvh.nodes[node.first].right_idx;

        const aabb_type& L_box = bvh.aabbs[L_idx];
        const aabb_type& R_box = bvh.aabbs[R_idx];

        const real_type L_mindist = mindist(L_box, q.target);
        const real_type R_mindist = mindist(R_box, q.target);

        const real_type L_minmaxdist = minmaxdist(L_box, q.target);
        const real_type R_minmaxdist = minmaxdist(R_box, q.target);

        // there should be an object that locates within minmaxdist.

        if(L_mindist <= R_minmaxdist) // L is worth considering
        {
            const auto obj_idx = bvh.nodes[L_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF) // leaf node
            {
                const real_type dist = calc_dist(q.target, bvh.objects[obj_idx]);
                if(dist <= dist_to_nearest_object)
                {
                    dist_to_nearest_object = dist;
                    nearest = obj_idx;
                }
            }
            else
            {
                *stack_ptr++ = thrust::make_pair(L_idx, L_mindist);
            }
        }
        if(R_mindist <= L_minmaxdist) // R is worth considering
        {
            const auto obj_idx = bvh.nodes[R_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF) // leaf node
            {
                const real_type dist = calc_dist(q.target, bvh.objects[obj_idx]);
                if(dist <= dist_to_nearest_object)
                {
                    dist_to_nearest_object = dist;
                    nearest = obj_idx;
                }
            }
            else
            {
                *stack_ptr++ = thrust::make_pair(R_idx, R_mindist);
            }
        }
        assert(stack_ptr < stack + 64);
    }
    while (stack < stack_ptr);
    return thrust::make_pair(nearest, dist_to_nearest_object);
}

} // lbvh
#endif// LBVH_QUERY_CUH
