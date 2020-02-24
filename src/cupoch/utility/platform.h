#pragma once
#include <cuda_runtime.h>
#include <iostream>

#ifdef _WIN32
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif
#if defined(__GNUC__)
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILENAME__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
    #define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILENAME__, __LINE__)
#endif

namespace cupoch {
namespace utility {

static const size_t MAX_NUM_STREAMS = 16;

cudaStream_t GetStream(size_t i);

int GetDevice();

void SetDevice(int device_no);

void Error(const char *error_string, const char *file, const int line,
           const char *func);
}
}

static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "") {
    if (cudaSuccess != err)
        cupoch::utility::Error(cudaGetErrorString(err), file, line, func);
}
