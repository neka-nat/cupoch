#include <thrust/host_vector.h>

#include "cupoch/geometry/voxelgrid.h"
#include "cupoch_pybind/device_map_wrapper.h"

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
    thrust::host_vector<KeyType> keys(other.size());
    thrust::host_vector<ValueType> values(other.size());
    size_t cnt = 0;
    for (const auto& it : other) {
        keys[cnt] = it.first;
        values[cnt] = it.second;
    }
    keys_ = keys;
    values_ = values;
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
    thrust::host_vector<KeyType> keys = keys_;
    thrust::host_vector<ValueType> values = values_;
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