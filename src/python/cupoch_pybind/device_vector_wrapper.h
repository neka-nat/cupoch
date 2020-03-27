#pragma once

#include <thrust/host_vector.h>

#include <Eigen/Core>

#include "cupoch/utility/device_vector.h"

namespace cupoch {

namespace geometry {
class PointCloud;
class TriangleMesh;
}  // namespace geometry

namespace wrapper {

template <typename Type>
class device_vector_wrapper {
public:
    device_vector_wrapper();
    device_vector_wrapper(const device_vector_wrapper<Type>& other);
    device_vector_wrapper(const thrust::host_vector<Type>& other);
    device_vector_wrapper(const utility::device_vector<Type>& other);
    device_vector_wrapper(utility::device_vector<Type>&& other) noexcept;
    ~device_vector_wrapper();
    device_vector_wrapper<Type>& operator=(
            const device_vector_wrapper<Type>& other);
    size_t size() const;
    bool empty() const;
    thrust::host_vector<Type> cpu() const;
    utility::device_vector<Type> data_;
};

using device_vector_vector3f = device_vector_wrapper<Eigen::Vector3f>;
using device_vector_vector2f = device_vector_wrapper<Eigen::Vector2f>;
using device_vector_vector3i = device_vector_wrapper<Eigen::Vector3i>;
using device_vector_vector2i = device_vector_wrapper<Eigen::Vector2i>;
using device_vector_int = device_vector_wrapper<int>;
using device_vector_size_t = device_vector_wrapper<size_t>;
using device_vector_float = device_vector_wrapper<float>;

template <typename Type>
void FromWrapper(utility::device_vector<Type>& dv,
                 const device_vector_wrapper<Type>& vec);

}  // namespace wrapper
}  // namespace cupoch