#ifndef LBVH_AABB_CUH
#define LBVH_AABB_CUH
#include "utility.cuh"
#include <thrust/swap.h>
#include <cmath>

namespace lbvh
{

template<typename T>
struct aabb
{
    typename vector_of<T>::type upper;
    typename vector_of<T>::type lower;
};

template<typename T>
__device__ __host__
inline bool intersects(const aabb<T>& lhs, const aabb<T>& rhs) noexcept
{
    if(lhs.upper.x < rhs.lower.x || rhs.upper.x < lhs.lower.x) {return false;}
    if(lhs.upper.y < rhs.lower.y || rhs.upper.y < lhs.lower.y) {return false;}
    if(lhs.upper.z < rhs.lower.z || rhs.upper.z < lhs.lower.z) {return false;}
    return true;
}

__device__ __host__
inline aabb<double> merge(const aabb<double>& lhs, const aabb<double>& rhs) noexcept
{
    aabb<double> merged;
    merged.upper.x = ::fmax(lhs.upper.x, rhs.upper.x);
    merged.upper.y = ::fmax(lhs.upper.y, rhs.upper.y);
    merged.upper.z = ::fmax(lhs.upper.z, rhs.upper.z);
    merged.lower.x = ::fmin(lhs.lower.x, rhs.lower.x);
    merged.lower.y = ::fmin(lhs.lower.y, rhs.lower.y);
    merged.lower.z = ::fmin(lhs.lower.z, rhs.lower.z);
    return merged;
}

__device__ __host__
inline aabb<float> merge(const aabb<float>& lhs, const aabb<float>& rhs) noexcept
{
    aabb<float> merged;
    merged.upper.x = ::fmaxf(lhs.upper.x, rhs.upper.x);
    merged.upper.y = ::fmaxf(lhs.upper.y, rhs.upper.y);
    merged.upper.z = ::fmaxf(lhs.upper.z, rhs.upper.z);
    merged.lower.x = ::fminf(lhs.lower.x, rhs.lower.x);
    merged.lower.y = ::fminf(lhs.lower.y, rhs.lower.y);
    merged.lower.z = ::fminf(lhs.lower.z, rhs.lower.z);
    return merged;
}

// metrics defined in
// Nearest Neighbor Queries (1995) ACS-SIGMOD
// - Nick Roussopoulos, Stephen Kelley FredericVincent

__device__ __host__
inline float mindist(const aabb<float>& lhs, const float4& rhs) noexcept
{
    const float dx = ::fminf(lhs.upper.x, ::fmaxf(lhs.lower.x, rhs.x)) - rhs.x;
    const float dy = ::fminf(lhs.upper.y, ::fmaxf(lhs.lower.y, rhs.y)) - rhs.y;
    const float dz = ::fminf(lhs.upper.z, ::fmaxf(lhs.lower.z, rhs.z)) - rhs.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ __host__
inline double mindist(const aabb<double>& lhs, const double4& rhs) noexcept
{
    const double dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
    const double dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
    const double dz = ::fmin(lhs.upper.z, ::fmax(lhs.lower.z, rhs.z)) - rhs.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ __host__
inline float minmaxdist(const aabb<float>& lhs, const float4& rhs) noexcept
{
    float3 rm_sq = make_float3((lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x),
                               (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y),
                               (lhs.lower.z - rhs.z) * (lhs.lower.z - rhs.z));
    float3 rM_sq = make_float3((lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x),
                               (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y),
                               (lhs.upper.z - rhs.z) * (lhs.upper.z - rhs.z));
    
    if((lhs.upper.x + lhs.lower.x) * 0.5f < rhs.x)
    {
        thrust::swap(rm_sq.x, rM_sq.x);
    }
    if((lhs.upper.y + lhs.lower.y) * 0.5f < rhs.y)
    {
        thrust::swap(rm_sq.y, rM_sq.y);
    }
    if((lhs.upper.z + lhs.lower.z) * 0.5f < rhs.z)
    {
        thrust::swap(rm_sq.z, rM_sq.z);
    }
    
    const float dx = rm_sq.x + rM_sq.y + rM_sq.z;
    const float dy = rM_sq.x + rm_sq.y + rM_sq.z;
    const float dz = rM_sq.x + rM_sq.y + rm_sq.z;
    return ::fminf(dx, ::fminf(dy, dz));
}

__device__ __host__
inline double minmaxdist(const aabb<double>& lhs, const double4& rhs) noexcept
{
    double3 rm_sq = make_double3((lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x),
                                 (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y),
                                 (lhs.lower.z - rhs.z) * (lhs.lower.z - rhs.z));
    double3 rM_sq = make_double3((lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x),
                                 (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y),
                                 (lhs.upper.z - rhs.z) * (lhs.upper.z - rhs.z));

    if((lhs.upper.x + lhs.lower.x) * 0.5 < rhs.x)
    {
        thrust::swap(rm_sq.x, rM_sq.x);
    }
    if((lhs.upper.y + lhs.lower.y) * 0.5 < rhs.y)
    {
        thrust::swap(rm_sq.y, rM_sq.y);
    }
    if((lhs.upper.z + lhs.lower.z) * 0.5 < rhs.z)
    {
        thrust::swap(rm_sq.z, rM_sq.z);
    }

    const double dx = rm_sq.x + rM_sq.y + rM_sq.z;
    const double dy = rM_sq.x + rm_sq.y + rM_sq.z;
    const double dz = rM_sq.x + rM_sq.y + rm_sq.z;
    return ::fmin(dx, ::fmin(dy, dz));
}

template<typename T>
__device__ __host__
inline typename vector_of<T>::type centroid(const aabb<T>& box) noexcept
{
    typename vector_of<T>::type c;
    c.x = (box.upper.x + box.lower.x) * 0.5;
    c.y = (box.upper.y + box.lower.y) * 0.5;
    c.z = (box.upper.z + box.lower.z) * 0.5;
    return c;
}

} // lbvh
#endif// LBVH_AABB_CUH
