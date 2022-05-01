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
#include "cupoch/utility/platform.h"
#if defined(__arm__) || defined(__aarch64__)
#include <GL/gl.h>
#endif
#include <cuda_gl_interop.h>

#include <mutex>

namespace cupoch {
namespace utility {

bool IsCudaAvailable() {
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    return num_gpus > 0;
}

cudaStream_t GetStream(size_t i) {
    static std::once_flag streamInitFlags[MAX_NUM_STREAMS];
    static cudaStream_t streams[MAX_NUM_STREAMS];
    std::call_once(streamInitFlags[i],
                   [i]() { cudaStreamCreate(&(streams[i])); });
    return streams[i];
}

int GetDevice() {
    int device_no;
    cudaGetDevice(&device_no);
    return device_no;
}

void SetDevice(int device_no) { cudaSetDevice(device_no); }

void Error(const char *error_string,
           const char *file,
           const int line,
           const char *func) {
    std::cout << "Error: " << error_string << "\t" << file << ":" << line
              << std::endl;
    exit(0);
}

}
}