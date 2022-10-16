#pragma once

namespace lbvh {
    struct AABB
    {
        float3 min;
        float3 max;
    };

    __device__
    inline float3 centroid(const AABB& box) noexcept
    {
        float3 c;
        c.x = (box.max.x + box.min.x) * 0.5;
        c.y = (box.max.y + box.min.y) * 0.5;
        c.z = (box.max.z + box.min.z) * 0.5;
        return c;
    }

//    __device__
//    inline AABB<double> merge(const AABB<double>& lhs, const AABB<double>& rhs) noexcept
//    {
//        AABB<double> merged;
//        merged.max.x = ::fmax(lhs.max.x, rhs.max.x);
//        merged.max.y = ::fmax(lhs.max.y, rhs.max.y);
//        merged.max.z = ::fmax(lhs.max.z, rhs.max.z);
//        merged.min.x = ::fmin(lhs.min.x, rhs.min.x);
//        merged.min.y = ::fmin(lhs.min.y, rhs.min.y);
//        merged.min.z = ::fmin(lhs.min.z, rhs.min.z);
//        return merged;
//    }

    __device__
    inline AABB merge(const AABB& lhs, const AABB& rhs) noexcept
    {
        AABB merged;
        merged.max.x = fmaxf(lhs.max.x, rhs.max.x);
        merged.max.y = fmaxf(lhs.max.y, rhs.max.y);
        merged.max.z = fmaxf(lhs.max.z, rhs.max.z);
        merged.min.x = fminf(lhs.min.x, rhs.min.x);
        merged.min.y = fminf(lhs.min.y, rhs.min.y);
        merged.min.z = fminf(lhs.min.z, rhs.min.z);
        return merged;
    }


    __forceinline__ __device__ float fmax3(float first, float second, float third) noexcept
    {
        return fmaxf(first, fmaxf(second, third));
    }

//    __forceinline__ __device__ float fmax3(double first, double second, double third) noexcept
//    {
//        return ::fmax(first, ::fmax(second, third));
//    }

    __forceinline__ __device__ float sq_length3(const float3& vec) noexcept {
        return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
    }

//    __forceinline__ __device__ double sq_length3(const vector_of_t<double>& vec) noexcept {
//        return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
//    }

    __forceinline__ __device__
    float dist_2_aabb(const float3& p, const AABB& aabb) noexcept
    {
        float sqDist(0);
        float v;

        if (p.x < aabb.min.x) v = aabb.min.x;
        if (p.x > aabb.max.x) v = aabb.max.x;
        if (p.x < aabb.min.x || p.x > aabb.max.x) sqDist += (v-p.x) * (v-p.x);

        if (p.y < aabb.min.y) v = aabb.min.y;
        if (p.y > aabb.max.y) v = aabb.max.y;
        if (p.y < aabb.min.y || p.y > aabb.max.y) sqDist += (v-p.y) * (v-p.y);

        if (p.z < aabb.min.z) v = aabb.min.z;
        if (p.z > aabb.max.z) v = aabb.max.z;
        if (p.z < aabb.min.z || p.z > aabb.max.z) sqDist += (v-p.z) * (v-p.z);
        return sqDist;
    }
}