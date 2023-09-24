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

void GetDeviceProp(cudaDeviceProp& prop, int device_no) {
    int cp_device_no = device_no;
    if (cp_device_no < 0) cp_device_no = GetDevice();
    cudaSafeCall(cudaGetDeviceProperties(&prop, cp_device_no));
}

void Error(const char *error_string,
           const char *file,
           const int line,
           const char *func) {
    std::cout << "Error: " << error_string << "\t" << file << ":" << line
              << std::endl;
    exit(0);
}

std::tuple<dim3, dim3> SelectBlockGridSizes(int data_size, int threads_per_block) {
    cudaDeviceProp prop;
    cupoch::utility::GetDeviceProp(prop);
    int max_threads_per_block = prop.maxThreadsPerBlock;
    if (threads_per_block > 0) {
        if (threads_per_block > max_threads_per_block) {
            throw std::runtime_error("Threads per block exceeds device maximum.");
        } else {
            max_threads_per_block = threads_per_block;
        }
    }
    int blocks_needed = static_cast<int>(std::ceil(data_size / float(max_threads_per_block)));
    if (blocks_needed <= prop.maxThreadsDim[0]) {
        return std::make_tuple(dim3(max_threads_per_block, 1, 1), dim3(blocks_needed, 1, 1));
    } else if (prop.maxThreadsDim[0] < blocks_needed) {
        blocks_needed = static_cast<int>(std::ceil(blocks_needed / float(prop.maxThreadsDim[0])));
        return std::make_tuple(dim3(max_threads_per_block, 1, 1), dim3(prop.maxThreadsDim[0], blocks_needed, 1));
    } else if (prop.maxThreadsDim[0] * prop.maxThreadsDim[1] < blocks_needed && blocks_needed <= prop.maxThreadsDim[0] * prop.maxThreadsDim[1] * prop.maxThreadsDim[2]) {
        blocks_needed = static_cast<int>(std::ceil(blocks_needed / float(prop.maxThreadsDim[0] * prop.maxThreadsDim[1])));
        return std::make_tuple(dim3(max_threads_per_block, 1, 1), dim3(prop.maxThreadsDim[0], prop.maxThreadsDim[1], blocks_needed));
    } else {
        throw std::runtime_error("Data size too large.");
    }
}

}
}