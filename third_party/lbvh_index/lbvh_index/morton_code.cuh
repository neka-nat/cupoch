#pragma once

#ifndef HASH_64
#define HASH_64 1 // just for debugging right now
#endif

namespace lbvh {
#if HASH_64
    typedef unsigned long long int HashType;

    __device__ __host__ inline HashType expand_bits(HashType v)
    {
        v = (v * 0x000100000001u) & 0xFFFF00000000FFFFu;
        v = (v * 0x000000010001u) & 0x00FF0000FF0000FFu;
        v = (v * 0x000000000101u) & 0xF00F00F00F00F00Fu;
        v = (v * 0x000000000011u) & 0x30C30C30C30C30C3u;
        v = (v * 0x000000000005u) & 0x9249249249249249u;
        return v;
    }
#else
    typedef unsigned int HashType;

    __device__ __host__ inline HashType expand_bits(HashType v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }
#endif

    // Calculates a Morton code for the
    // given 3D point located within the unit cube [0,1].
    __device__ __host__
    inline HashType morton_code(float3 xyz, float resolution = 1024.0f) noexcept
    {
#if HASH_64
        resolution *= resolution; // increase the resolution for 64 bit codes
#endif
        xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
        xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
        xyz.z = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
        const HashType xx = expand_bits(static_cast<HashType>(xyz.x));
        const HashType yy = expand_bits(static_cast<HashType>(xyz.y));
        const HashType zz = expand_bits(static_cast<HashType>(xyz.z));
        return xx * 4 + yy * 2 + zz;
    }

//    __device__ __host__
//    inline HashType morton_code(double3 xyz, double resolution = 1024.0) noexcept
//    {
//#if HASH_64
//        resolution *= resolution;
//#endif
//        xyz.x = ::fmin(::fmax(xyz.x * resolution, 0.0), resolution - 1.0);
//        xyz.y = ::fmin(::fmax(xyz.y * resolution, 0.0), resolution - 1.0);
//        xyz.z = ::fmin(::fmax(xyz.z * resolution, 0.0), resolution - 1.0);
//        const HashType xx = expand_bits(static_cast<HashType>(xyz.x));
//        const HashType yy = expand_bits(static_cast<HashType>(xyz.y));
//        const HashType zz = expand_bits(static_cast<HashType>(xyz.z));
//        return xx * 4 + yy * 2 + zz;
//    }


    // Returns the highest differing bit of the two morton codes
    __device__ inline int highest_bit(HashType lhs, HashType rhs) noexcept
    {
        return lhs ^ rhs;
    }
//#if HASH_64
//    __device__
//    inline int common_upper_bits(const HashType lhs, const HashType rhs) noexcept
//    {
//        return ::__clzll(lhs ^ rhs);
//    }
//#else
//    __device__
//    inline int common_upper_bits(const HashType lhs, const HashType rhs) noexcept
//    {
//        return ::__clz(lhs ^ rhs);
//    }
//#endif
}