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