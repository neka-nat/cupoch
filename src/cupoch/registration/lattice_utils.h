/**
 * Copyright (c) 2020 Neka-Nat
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
#pragma once

#include <cuda_runtime.h>

#include <Eigen/Core>

namespace cupoch {
namespace registration {

template <int Dim>
struct LatticeCoordKey {
    __host__ __device__ LatticeCoordKey(){};
    __host__ __device__ ~LatticeCoordKey(){};

    // The hashing of this key
    __host__ __device__ __forceinline__ unsigned short hash() const {
        unsigned short hash_value = 0;
        for (int i = 0; i < Dim; i++) {
            hash_value += key_[i];
            hash_value *= 1500007;  // This is a prime number
        }
        return hash_value;
    }

    // The comparator of a key
    __host__ __device__ __forceinline__ char less_than(
            const LatticeCoordKey<Dim>& rhs) const {
        char is_less_than = 0;
        for (int i = 0; i < Dim; i++) {
            if (key_[i] < rhs.key_[i]) {
                is_less_than = 1;
                break;
            } else if (key_[i] > rhs.key_[i]) {
                is_less_than = -1;
                break;
            }
        }
        return is_less_than;
    }

    // Operator
    __host__ __device__ __forceinline__ bool operator==(
            const LatticeCoordKey<Dim>& rhs) const {
        for (int i = 0; i < Dim; i++) {
            if (key_[i] != rhs.key_[i]) return false;
        }
        return true;
    }

    Eigen::Matrix<short, Dim, 1> key_;
};

///
/// \brief Compute the lattice key and the weight of the lattice point
///        surround this feature.
/// \tparam Dim
/// \param feature The feature vector, in the size of Dim
/// \param lattice_coord_keys The lattice coord keys nearby this feature. The
///                           array is in the size of Dim + 1.
/// \param barycentric The weight value, in the size of Dim + 2, while
///                    the first Dim + 1 elements match the weight
///                    of the lattice_coord_keys
/// \param no_blur The flag of calculating without or with blur
///
template <int Dim>
__host__ __device__ __forceinline__ void CreateLatticeGrid(
        float* feature,
        LatticeCoordKey<Dim>* lattice_coord_keys,
        float* barycentric,
        bool no_blur = false);

}  // namespace registration
}  // namespace cupoch

#include "cupoch/registration/lattice_utils.inl"