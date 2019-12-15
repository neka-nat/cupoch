#include "cupoch/utility/global_stream.h"
#include <mutex>

using namespace cupoch;
using namespace cupoch::utility;

cudaStream_t cupoch::utility::GetGlobalStream(size_t i) {
    static std::once_flag streamInitFlags[MAX_DEVICES];
    static cudaStream_t streams[MAX_DEVICES];
    std::call_once(streamInitFlags[i], [i]() {
        cudaStreamCreate(&(streams[i]));
    });
    return streams[i];
}