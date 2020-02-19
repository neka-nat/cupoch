#include "cupoch/utility/dl_converter.h"
#include "cupoch/utility/platform.h"
#include <type_traits>

using namespace cupoch;
using namespace cupoch::utility;

namespace {

template<typename T>
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

template<typename T, int Dim>
struct DeviceVectorDLMTensor {
    thrust::device_vector<Eigen::Matrix<T, Dim, 1>> handle;
    DLManagedTensor tensor;
};

template<typename T, int Dim>
void deleter(DLManagedTensor* arg) {
    delete[] arg->dl_tensor.shape;
    delete static_cast<DeviceVectorDLMTensor<T, Dim>*>(arg->manager_ctx);
}

}

template<typename T, int Dim>
DLManagedTensor* cupoch::utility::ToDLPack(const utility::device_vector<Eigen::Matrix<T, Dim, 1>>& src) {
    DeviceVectorDLMTensor<T, Dim>* dvdl(new DeviceVectorDLMTensor<T, Dim>);
    dvdl->handle = src;
    dvdl->tensor.manager_ctx = dvdl;
    dvdl->tensor.deleter = &deleter<T, Dim>;
    dvdl->tensor.dl_tensor.data = const_cast<void*>((const void*)(thrust::raw_pointer_cast(src.data())));
    int64_t device_id = GetDevice();
    DLContext ctx;
    ctx.device_id = device_id;
    ctx.device_type = DLDeviceType::kDLGPU;
    dvdl->tensor.dl_tensor.ctx = ctx;
    dvdl->tensor.dl_tensor.ndim = 2;
    DLDataType dtype;
    dtype.lanes = 1;
    dtype.bits = sizeof(T) * 8;
    dtype.code = GetDLDataTypeCode<T>();
    dvdl->tensor.dl_tensor.dtype = dtype;
    int64_t* shape = new int64_t[2];
    shape[0] = src.size();
    shape[1] = Dim;
    dvdl->tensor.dl_tensor.shape = shape;
    dvdl->tensor.dl_tensor.strides = nullptr;
    dvdl->tensor.dl_tensor.byte_offset = 0;
    return &(dvdl->tensor);
}

template DLManagedTensor* cupoch::utility::ToDLPack(const utility::device_vector<Eigen::Matrix<float, 3, 1>>& src);
template DLManagedTensor* cupoch::utility::ToDLPack(const utility::device_vector<Eigen::Matrix<int, 2, 1>>& src);
template DLManagedTensor* cupoch::utility::ToDLPack(const utility::device_vector<Eigen::Matrix<int, 3, 1>>& src);

template<>
void cupoch::utility::FromDLPack(const DLManagedTensor* src, utility::device_vector<Eigen::Matrix<float, 3, 1>>& dst) {
    dst.resize(src->dl_tensor.shape[0]);
    auto base_ptr = thrust::device_pointer_cast((Eigen::Matrix<float, 3, 1>*)src->dl_tensor.data);
    thrust::copy(base_ptr, base_ptr + src->dl_tensor.shape[0], dst.begin());
}

template<>
void cupoch::utility::FromDLPack(const DLManagedTensor* src, utility::device_vector<Eigen::Matrix<int, 2, 1>>& dst) {
    dst.resize(src->dl_tensor.shape[0]);
    auto base_ptr = thrust::device_pointer_cast((Eigen::Matrix<int, 2, 1>*)src->dl_tensor.data);
    thrust::copy(base_ptr, base_ptr + src->dl_tensor.shape[0], dst.begin());
}

template<>
void cupoch::utility::FromDLPack(const DLManagedTensor* src, utility::device_vector<Eigen::Matrix<int, 3, 1>>& dst) {
    dst.resize(src->dl_tensor.shape[0]);
    auto base_ptr = thrust::device_pointer_cast((Eigen::Matrix<int, 3, 1>*)src->dl_tensor.data);
    thrust::copy(base_ptr, base_ptr + src->dl_tensor.shape[0], dst.begin());
}