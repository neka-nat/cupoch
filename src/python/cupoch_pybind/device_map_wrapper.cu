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
#include <thrust/host_vector.h>

#include "cupoch/geometry/voxelgrid.h"
#include "cupoch_pybind/device_map_wrapper.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace wrapper {

template <typename KeyType, typename ValueType, typename Hash>
device_map_wrapper<KeyType, ValueType, Hash>::device_map_wrapper(){};
template <typename KeyType, typename ValueType, typename Hash>
device_map_wrapper<KeyType, ValueType, Hash>::device_map_wrapper(
        const device_map_wrapper<KeyType, ValueType, Hash>& other)
    : keys_(other.keys_), values_(other.values_) {}
template <typename KeyType, typename ValueType, typename Hash>
device_map_wrapper<KeyType, ValueType, Hash>::device_map_wrapper(
        const std::unordered_map<KeyType, ValueType, Hash>& other) {
    utility::pinned_host_vector<KeyType> keys(other.size());
    utility::pinned_host_vector<ValueType> values(other.size());
    size_t cnt = 0;
    for (const auto& it : other) {
        keys[cnt] = it.first;
        values[cnt] = it.second;
    }
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(keys_.data()), thrust::raw_pointer_cast(keys.data()),
                            other.size() * sizeof(KeyType), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(values_.data()), thrust::raw_pointer_cast(values.data()),
                            other.size() * sizeof(ValueType), cudaMemcpyHostToDevice));
}

template <typename KeyType, typename ValueType, typename Hash>
device_map_wrapper<KeyType, ValueType, Hash>::device_map_wrapper(
        const utility::device_vector<KeyType>& key_other,
        const utility::device_vector<ValueType>& value_other)
    : keys_(key_other), values_(value_other) {}

template <typename KeyType, typename ValueType, typename Hash>
device_map_wrapper<KeyType, ValueType, Hash>::~device_map_wrapper(){};

template <typename KeyType, typename ValueType, typename Hash>
device_map_wrapper<KeyType, ValueType, Hash>&
device_map_wrapper<KeyType, ValueType, Hash>::operator=(
        const device_map_wrapper<KeyType, ValueType, Hash>& other) {
    keys_ = other.keys_;
    values_ = other.values_;
    return *this;
}

template <typename KeyType, typename ValueType, typename Hash>
size_t device_map_wrapper<KeyType, ValueType, Hash>::size() const {
    return keys_.size();
}

template <typename KeyType, typename ValueType, typename Hash>
bool device_map_wrapper<KeyType, ValueType, Hash>::empty() const {
    return keys_.empty();
}

template <typename KeyType, typename ValueType, typename Hash>
std::unordered_map<KeyType, ValueType, Hash>
device_map_wrapper<KeyType, ValueType, Hash>::cpu() const {
    utility::pinned_host_vector<KeyType> keys(keys_.size());
    utility::pinned_host_vector<ValueType> values(values_.size());
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(keys.data()), thrust::raw_pointer_cast(keys_.data()),
                            keys.size() * sizeof(KeyType), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(values.data()), thrust::raw_pointer_cast(values_.data()),
                            values.size() * sizeof(ValueType), cudaMemcpyDeviceToHost));
    std::unordered_map<KeyType, ValueType, Hash> ans;
    for (int i = 0; i < keys.size(); i++) {
        ans[keys[i]] = values[i];
    }
    return ans;
}

template <typename KeyType, typename ValueType, typename Hash>
void FromWrapper(utility::device_vector<KeyType>& dk,
                 utility::device_vector<ValueType>& dv,
                 const device_map_wrapper<KeyType, ValueType, Hash>& map) {
    dk = map.keys_;
    dv = map.values_;
}

template class device_map_wrapper<Eigen::Vector3i,
                                  geometry::Voxel,
                                  hash<Eigen::Vector3i>>;
template void
FromWrapper<Eigen::Vector3i, geometry::Voxel, hash<Eigen::Vector3i>>(
        utility::device_vector<Eigen::Vector3i>& dk,
        utility::device_vector<geometry::Voxel>& dv,
        const device_map_wrapper<Eigen::Vector3i,
                                 geometry::Voxel,
                                 hash<Eigen::Vector3i>>& vec);

}  // namespace wrapper
}  // namespace cupoch