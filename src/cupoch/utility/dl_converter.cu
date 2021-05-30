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
#include <type_traits>

#include "cupoch/utility/console.h"
#include "cupoch/utility/dl_converter.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace utility {

namespace {

template <typename T>
DLDataTypeCode GetDLDataTypeCode() {
    if (std::is_same<T, int>::value) {
        return DLDataTypeCode::kDLInt;
    } else if (std::is_same<T, char>::value) {
        return DLDataTypeCode::kDLInt;
    } else if (std::is_same<T, float>::value) {
        return DLDataTypeCode::kDLFloat;
    } else if (std::is_same<T, double>::value) {
        return DLDataTypeCode::kDLFloat;
    } else {
        throw std::logic_error("Invalid data type.");
    }
}

template <typename T, int Dim>
struct DeviceVectorDLMTensor {
    thrust::device_vector<Eigen::Matrix<T, Dim, 1>> handle;
    DLManagedTensor tensor;
};

template <typename T, int Dim>
void deleter(DLManagedTensor *arg) {
    delete[] arg->dl_tensor.shape;
    delete static_cast<DeviceVectorDLMTensor<T, Dim> *>(arg->manager_ctx);
}

}  // namespace

template <typename T, int Dim>
DLManagedTensor *ToDLPack(
        const utility::device_vector<Eigen::Matrix<T, Dim, 1>> &src) {
    DeviceVectorDLMTensor<T, Dim> *dvdl(new DeviceVectorDLMTensor<T, Dim>);
    dvdl->handle = src;
    dvdl->tensor.manager_ctx = dvdl;
    dvdl->tensor.deleter = &deleter<T, Dim>;
    dvdl->tensor.dl_tensor.data = const_cast<void *>(
            (const void *)(thrust::raw_pointer_cast(src.data())));
    int64_t device_id = GetDevice();
    DLContext device;
    device.device_id = device_id;
    device.device_type = DLDeviceType::kDLGPU;
    dvdl->tensor.dl_tensor.device = device;
    dvdl->tensor.dl_tensor.ndim = 2;
    DLDataType dtype;
    dtype.lanes = 1;
    dtype.bits = sizeof(T) * 8;
    dtype.code = GetDLDataTypeCode<T>();
    dvdl->tensor.dl_tensor.dtype = dtype;
    int64_t *shape = new int64_t[2];
    shape[0] = src.size();
    shape[1] = Dim;
    dvdl->tensor.dl_tensor.shape = shape;
    dvdl->tensor.dl_tensor.strides = nullptr;
    dvdl->tensor.dl_tensor.byte_offset = 0;
    return &(dvdl->tensor);
}

template DLManagedTensor *ToDLPack(
        const utility::device_vector<Eigen::Matrix<float, 3, 1>> &src);
template DLManagedTensor *ToDLPack(
        const utility::device_vector<Eigen::Matrix<float, 2, 1>> &src);
template DLManagedTensor *ToDLPack(
        const utility::device_vector<Eigen::Matrix<float, 1, 1>> &src);
template DLManagedTensor *ToDLPack(
        const utility::device_vector<Eigen::Matrix<int, 2, 1>> &src);
template DLManagedTensor *ToDLPack(
        const utility::device_vector<Eigen::Matrix<int, 3, 1>> &src);

template <typename T, int Dim>
void FromDLPack(const DLManagedTensor *src,
                utility::device_vector<Eigen::Matrix<T, Dim, 1>> &dst) {
    dst.resize(src->dl_tensor.shape[0]);
    auto base_ptr = thrust::device_pointer_cast(
            (Eigen::Matrix<T, Dim, 1> *)src->dl_tensor.data);
    if (src->dl_tensor.device.device_type == DLDeviceType::kDLCPU) {
        cudaSafeCall(cudaMemcpy(
                thrust::raw_pointer_cast(dst.data()),
                thrust::raw_pointer_cast(base_ptr),
                src->dl_tensor.shape[0] * sizeof(Eigen::Matrix<T, Dim, 1>),
                cudaMemcpyHostToDevice));
    } else if (src->dl_tensor.device.device_type == DLDeviceType::kDLGPU) {
        thrust::copy(base_ptr, base_ptr + src->dl_tensor.shape[0], dst.begin());
    } else {
        utility::LogError("[FromDLPack] Unsupported device type.");
    }
}

template void FromDLPack(
        const DLManagedTensor *src,
        utility::device_vector<Eigen::Matrix<float, 3, 1>> &dst);
template void FromDLPack(
        const DLManagedTensor *src,
        utility::device_vector<Eigen::Matrix<float, 2, 1>> &dst);
template void FromDLPack(
        const DLManagedTensor *src,
        utility::device_vector<Eigen::Matrix<float, 1, 1>> &dst);
template void FromDLPack(const DLManagedTensor *src,
                         utility::device_vector<Eigen::Matrix<int, 2, 1>> &dst);
template void FromDLPack(const DLManagedTensor *src,
                         utility::device_vector<Eigen::Matrix<int, 3, 1>> &dst);

}  // namespace utility
}  // namespace cupoch