#include "cupoch/utility/platform.h"
#include <mutex>

using namespace cupoch;
using namespace cupoch::utility;

cudaStream_t cupoch::utility::GetStream(size_t i) {
    static std::once_flag streamInitFlags[MAX_DEVICES];
    static cudaStream_t streams[MAX_DEVICES];
    std::call_once(streamInitFlags[i], [i]() {
        cudaStreamCreate(&(streams[i]));
    });
    return streams[i];
}

int cupoch::utility::GetDevice() {
    int device_no;
    cudaGetDevice(&device_no);
    return device_no;
}

void cupoch::utility::SetDevice(int device_no) {
    cudaSetDevice(device_no);
}