#include "cupoch/utility/dl_converter.h"
#include "cupoch/utility/platform.h"
#include <dlpack/contrib/dlpack/dlpackcpp.h>

using namespace cupoch;
using namespace cupoch::utility;

void cupoch::utility::ToDLPack(const thrust::device_vector<Eigen::Vector3f>& src, dlpack::DLTContainer& dst) {
    auto handle = DLTensor(dst);
    handle.data = const_cast<void*>((const void*)(thrust::raw_pointer_cast(src.data())));
    int64_t device_id = GetDevice();
    handle.ctx.device_id = device_id;
    handle.ctx.device_type = DLDeviceType::kDLGPU;
    handle.ndim = 2;
    handle.dtype.lanes = 1U;
    handle.dtype.bits = sizeof(float) * 8;
    handle.dtype.code = DLDataTypeCode::kDLFloat;
    dst.Reshape({(int64_t)(src.size()), 3});
}

void cupoch::utility::FromDLPack(const dlpack::DLTContainer& src, const thrust::device_vector<Eigen::Vector3f>& dst) {
}
