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

#include <Eigen/Core>
#include <unordered_map>

#include "cupoch/utility/device_vector.h"

namespace cupoch {

namespace geometry {
class Voxel;
class OccupancyVoxel;
}  // namespace geometry

namespace wrapper {

template <typename T>
struct hash {
    std::size_t operator()(T const& matrix) const {
        size_t seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                    (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

template <typename KeyType,
          typename ValueType,
          typename Hash = std::hash<KeyType>>
class device_map_wrapper {
public:
    device_map_wrapper();
    device_map_wrapper(
            const device_map_wrapper<KeyType, ValueType, Hash>& other);
    device_map_wrapper(
            const std::unordered_map<KeyType, ValueType, Hash>& other);
    device_map_wrapper(const utility::device_vector<KeyType>& key_other,
                       const utility::device_vector<ValueType>& value_other);
    ~device_map_wrapper();
    device_map_wrapper<KeyType, ValueType, Hash>& operator=(
            const device_map_wrapper<KeyType, ValueType, Hash>& other);
    size_t size() const;
    bool empty() const;
    std::unordered_map<KeyType, ValueType, Hash> cpu() const;
    utility::device_vector<KeyType> keys_;
    utility::device_vector<ValueType> values_;
};

template <typename KeyType,
          typename ValueType,
          typename Hash = std::hash<KeyType>>
void FromWrapper(utility::device_vector<KeyType>& dk,
                 utility::device_vector<ValueType>& dv,
                 const device_map_wrapper<KeyType, ValueType, Hash>& map);

using VoxelMap = device_map_wrapper<Eigen::Vector3i,
                                    geometry::Voxel,
                                    hash<Eigen::Vector3i>>;

}  // namespace wrapper
}  // namespace cupoch