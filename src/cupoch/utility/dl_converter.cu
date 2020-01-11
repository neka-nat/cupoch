#include "cupoch/utility/dl_converter.h"
#include "cupoch/utility/platform.h"
#include <dlpack/contrib/dlpack/dlpackcpp.h>

using namespace cupoch;
using namespace cupoch::utility;
namespace {
struct DeviceVectorDLMTensor {
    thrust::device_vector<Eigen::Vector3f> handle;
    DLManagedTensor tensor;
};

void deleter(DLManagedTensor* arg) {
    delete static_cast<DeviceVectorDLMTensor*>(arg->manager_ctx);
}

}


void cupoch::utility::ToDLPack(const thrust::device_vector<Eigen::Vector3f>& src, DLManagedTensor** dst) {
    DeviceVectorDLMTensor* dvdl(new DeviceVectorDLMTensor);
    dvdl->handle = src;
    dvdl->tensor.manager_ctx = dvdl;
    dvdl->tensor.deleter = &deleter;
    dvdl->tensor.dl_tensor.data = const_cast<void*>((const void*)(thrust::raw_pointer_cast(src.data())));
    int64_t device_id = GetDevice();
    DLContext ctx;
    ctx.device_id = device_id;
    ctx.device_type = DLDeviceType::kDLGPU;
    dvdl->tensor.dl_tensor.ctx = ctx;
    dvdl->tensor.dl_tensor.ndim = 2;
    DLDataType dtype;
    dtype.lanes = 1;
    dtype.bits = sizeof(float) * 8;
    dtype.code = DLDataTypeCode::kDLFloat;
    dvdl->tensor.dl_tensor.dtype = dtype;
    int64_t* shape = new int64_t[2];
    shape[0] = src.size();
    shape[1] = 3;
    int64_t* strides = new int64_t[2];
    shape[0] = sizeof(float) * 3;
    shape[1] = sizeof(float);
    dvdl->tensor.dl_tensor.shape = shape;
    dvdl->tensor.dl_tensor.strides = strides;
    dvdl->tensor.dl_tensor.byte_offset = 0;
    *dst = &(dvdl->tensor);
}

void cupoch::utility::FromDLPack(const DLManagedTensor* src, const thrust::device_vector<Eigen::Vector3f>& dst) {
}
