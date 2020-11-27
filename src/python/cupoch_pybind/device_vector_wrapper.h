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

#include "cupoch/collision/primitives.h"
#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/utility/device_vector.h"

namespace cupoch {

namespace wrapper {

template <typename Type>
class device_vector_wrapper {
public:
    device_vector_wrapper();
    device_vector_wrapper(const device_vector_wrapper<Type>& other);
    device_vector_wrapper(const utility::pinned_host_vector<Type>& other);
    device_vector_wrapper(const utility::device_vector<Type>& other);
    device_vector_wrapper(utility::device_vector<Type>&& other) noexcept;
    ~device_vector_wrapper();
    device_vector_wrapper<Type>& operator=(
            const device_vector_wrapper<Type>& other);
    device_vector_wrapper<Type>& operator+=(
            const utility::device_vector<Type>& other);
    device_vector_wrapper<Type>& operator+=(
            const thrust::host_vector<Type>& other);
    device_vector_wrapper<Type>& operator-=(
            const utility::device_vector<Type>& other);
    device_vector_wrapper<Type>& operator-=(
            const thrust::host_vector<Type>& other);
    size_t size() const;
    bool empty() const;
    void push_back(const Type& x);
    utility::pinned_host_vector<Type> cpu() const;
    utility::device_vector<Type> data_;
};

template <typename Type>
device_vector_wrapper<Type> operator+ (const device_vector_wrapper<Type>& lhs, const device_vector_wrapper<Type>& rhs) {
    device_vector_wrapper<Type> ans = lhs;
    ans += rhs;
    return ans;
}

template <typename Type>
device_vector_wrapper<Type> operator- (const device_vector_wrapper<Type>& lhs, const device_vector_wrapper<Type>& rhs) {
    device_vector_wrapper<Type> ans = lhs;
    ans -= rhs;
    return ans;
}

using device_vector_vector3f = device_vector_wrapper<Eigen::Vector3f>;
using device_vector_vector2f = device_vector_wrapper<Eigen::Vector2f>;
using device_vector_vector3i = device_vector_wrapper<Eigen::Vector3i>;
using device_vector_vector2i = device_vector_wrapper<Eigen::Vector2i>;
using device_vector_vector33f =
        device_vector_wrapper<Eigen::Matrix<float, 33, 1>>;
using device_vector_int = device_vector_wrapper<int>;
using device_vector_size_t = device_vector_wrapper<size_t>;
using device_vector_float = device_vector_wrapper<float>;
using device_vector_occupancyvoxel =
        device_vector_wrapper<geometry::OccupancyVoxel>;
using device_vector_primitives =
        device_vector_wrapper<collision::PrimitivePack>;

#if defined(_WIN32)
using device_vector_ulong = device_vector_wrapper<unsigned long>;
#endif

template <typename Type>
void FromWrapper(utility::device_vector<Type>& dv,
                 const device_vector_wrapper<Type>& vec);

}  // namespace wrapper
}  // namespace cupoch