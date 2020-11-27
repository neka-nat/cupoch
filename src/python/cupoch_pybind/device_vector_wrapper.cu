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
#include "cupoch_pybind/device_vector_wrapper.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace wrapper {

template <typename Type>
device_vector_wrapper<Type>::device_vector_wrapper(){};
template <typename Type>
device_vector_wrapper<Type>::device_vector_wrapper(
        const device_vector_wrapper<Type>& other)
    : data_(other.data_) {}
template <typename Type>
device_vector_wrapper<Type>::device_vector_wrapper(
        const utility::pinned_host_vector<Type>& other)
    : data_(other) {}
template <typename Type>
device_vector_wrapper<Type>::device_vector_wrapper(
        const utility::device_vector<Type>& other)
    : data_(other) {}
template <typename Type>
device_vector_wrapper<Type>::device_vector_wrapper(
        utility::device_vector<Type>&& other) noexcept
    : data_(std::move(other)) {}
template <typename Type>
device_vector_wrapper<Type>::~device_vector_wrapper(){};

template <typename Type>
device_vector_wrapper<Type>& device_vector_wrapper<Type>::operator=(
        const device_vector_wrapper<Type>& other) {
    data_ = other.data_;
    return *this;
}

template <typename Type>
device_vector_wrapper<Type>& device_vector_wrapper<Type>::operator+=(
        const utility::device_vector<Type>& other) {
    thrust::transform(data_.begin(), data_.end(), other.begin(), data_.begin(),
                      thrust::plus<Type>());
    return *this;
}

template <typename Type>
device_vector_wrapper<Type>& device_vector_wrapper<Type>::operator+=(
        const thrust::host_vector<Type>& other) {
    utility::device_vector<Type> dvo = other;
    thrust::transform(data_.begin(), data_.end(), dvo.begin(), data_.begin(),
                      thrust::plus<Type>());
    return *this;
}

template <typename Type>
device_vector_wrapper<Type>& device_vector_wrapper<Type>::operator-=(
        const utility::device_vector<Type>& other) {
    thrust::transform(data_.begin(), data_.end(), other.begin(), data_.begin(),
                      thrust::minus<Type>());
    return *this;
}

template <typename Type>
device_vector_wrapper<Type>& device_vector_wrapper<Type>::operator-=(
        const thrust::host_vector<Type>& other) {
    utility::device_vector<Type> dvo = other;
    thrust::transform(data_.begin(), data_.end(), dvo.begin(), data_.begin(),
                      thrust::minus<Type>());
    return *this;
}

template <typename Type>
size_t device_vector_wrapper<Type>::size() const {
    return data_.size();
}

template <typename Type>
bool device_vector_wrapper<Type>::empty() const {
    return data_.empty();
}

template <typename Type>
void device_vector_wrapper<Type>::push_back(const Type& x) {
    data_.push_back(x);
}

template <typename Type>
utility::pinned_host_vector<Type> device_vector_wrapper<Type>::cpu() const {
    utility::pinned_host_vector<Type> ans(data_.size());
    cudaSafeCall(cudaMemcpy(ans.data(), thrust::raw_pointer_cast(data_.data()), sizeof(Type) * data_.size(), cudaMemcpyDeviceToHost));
    return ans;
}

template class device_vector_wrapper<Eigen::Vector3f>;
template class device_vector_wrapper<Eigen::Vector2f>;
template class device_vector_wrapper<Eigen::Vector3i>;
template class device_vector_wrapper<Eigen::Vector2i>;
template class device_vector_wrapper<Eigen::Matrix<float, 33, 1>>;
template class device_vector_wrapper<float>;
template class device_vector_wrapper<int>;
template class device_vector_wrapper<size_t>;
template class device_vector_wrapper<geometry::OccupancyVoxel>;
template class device_vector_wrapper<collision::PrimitivePack>;

template <typename Type>
void FromWrapper(utility::device_vector<Type>& dv,
                 const device_vector_wrapper<Type>& vec) {
    dv = vec.data_;
}

template void FromWrapper<Eigen::Vector3f>(
        utility::device_vector<Eigen::Vector3f>& dv,
        const device_vector_wrapper<Eigen::Vector3f>& vec);
template void FromWrapper<Eigen::Vector2f>(
        utility::device_vector<Eigen::Vector2f>& dv,
        const device_vector_wrapper<Eigen::Vector2f>& vec);
template void FromWrapper<Eigen::Vector3i>(
        utility::device_vector<Eigen::Vector3i>& dv,
        const device_vector_wrapper<Eigen::Vector3i>& vec);
template void FromWrapper<Eigen::Vector2i>(
        utility::device_vector<Eigen::Vector2i>& dv,
        const device_vector_wrapper<Eigen::Vector2i>& vec);
template void FromWrapper<Eigen::Matrix<float, 33, 1>>(
        utility::device_vector<Eigen::Matrix<float, 33, 1>>& dv,
        const device_vector_wrapper<Eigen::Matrix<float, 33, 1>>& vec);
template void FromWrapper<float>(utility::device_vector<float>& dv,
                                 const device_vector_wrapper<float>& vec);
template void FromWrapper<int>(utility::device_vector<int>& dv,
                               const device_vector_wrapper<int>& vec);
template void FromWrapper<size_t>(utility::device_vector<size_t>& dv,
                                  const device_vector_wrapper<size_t>& vec);
template void FromWrapper<geometry::OccupancyVoxel>(
        utility::device_vector<geometry::OccupancyVoxel>& dv,
        const device_vector_wrapper<geometry::OccupancyVoxel>& vec);
template void FromWrapper<collision::PrimitivePack>(
        utility::device_vector<collision::PrimitivePack>& dv,
        const device_vector_wrapper<collision::PrimitivePack>& vec);

#if defined(_WIN32)
template class device_vector_wrapper<unsigned long>;
template void FromWrapper<unsigned long>(
    utility::device_vector<unsigned long>& dv,
    const device_vector_wrapper<unsigned long>& vec);
#endif
}  // namespace wrapper
}  // namespace cupoch