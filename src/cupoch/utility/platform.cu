#include "cupoch/utility/platform.h"
#if defined(__arm__) || defined(__aarch64__)
#include <GL/gl.h>
#endif
#include <cuda_gl_interop.h>
#include <mutex>

using namespace cupoch;
using namespace cupoch::utility;

cudaStream_t cupoch::utility::GetStream(size_t i) {
    static std::once_flag streamInitFlags[MAX_NUM_STREAMS];
    static cudaStream_t streams[MAX_NUM_STREAMS];
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

void cupoch::utility::Error(const char *error_string, const char *file, const int line,
                            const char *func) {
    std::cout << "Error: " << error_string << "\t" << file << ":" << line
              << std::endl;
    exit(0);
}