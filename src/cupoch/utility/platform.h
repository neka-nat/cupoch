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
#if defined(_WIN32)
#include <windows.h>
#endif
#include <cuda_runtime.h>

#include <iostream>

#ifdef _WIN32
#define __FILENAME__ \
    (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__ \
    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif
#if defined(__GNUC__)
#define cudaSafeCall(expr) \
    ___cudaSafeCall(expr, __FILENAME__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
#define cudaSafeCall(expr) ___cudaSafeCall(expr, __FILENAME__, __LINE__)
#endif

namespace cupoch {
namespace utility {

static const size_t MAX_NUM_STREAMS = 16;

cudaStream_t GetStream(size_t i);

int GetDevice();

void SetDevice(int device_no);

void Error(const char *error_string,
           const char *file,
           const int line,
           const char *func);
}  // namespace utility
}  // namespace cupoch

static inline void ___cudaSafeCall(cudaError_t err,
                                   const char *file,
                                   const int line,
                                   const char *func = "") {
    if (cudaSuccess != err)
        cupoch::utility::Error(cudaGetErrorString(err), file, line, func);
}
