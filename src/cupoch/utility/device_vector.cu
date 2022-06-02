/**
 * Copyright (c) 2022 Neka-Nat
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
#include "cupoch/utility/device_vector.h"

namespace cupoch {
namespace utility {

#ifdef USE_RMM

void InitializeAllocator(
        rmmAllocationMode_t mode,
        size_t initial_pool_size,
        const std::vector<int> &devices) {
    static std::vector<std::shared_ptr<rmm::mr::device_memory_resource>> per_device_memory = {};
    static std::vector<int> s_devices = {};
    static bool is_initialized = false;

    if (is_initialized) {
        rmm::mr::set_per_device_resource(rmm::cuda_device_id{0}, nullptr);
        for (auto d: s_devices) {
            rmm::mr::set_per_device_resource(rmm::cuda_device_id{d}, nullptr);
        }
        s_devices.clear();
        per_device_memory.clear();
    }
    s_devices = devices;
    if (s_devices.empty()) s_devices.push_back(0);

    for (auto d: s_devices) {
        cudaSetDevice(d);
        if (mode & CudaManagedMemory) {
            auto cuda_mr = std::make_shared<rmm::mr::managed_memory_resource>();
            if (mode & PoolAllocation) {
                auto pool = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(cuda_mr, initial_pool_size);
                per_device_memory.emplace_back(pool);
            } else {
                per_device_memory.emplace_back(cuda_mr);
            }
        } else {
            auto cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
            if (mode & PoolAllocation) {
                auto pool = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(cuda_mr, initial_pool_size);
                per_device_memory.emplace_back(pool);
            } else {
                per_device_memory.emplace_back(cuda_mr);
            }
        }
        rmm::mr::set_per_device_resource(rmm::cuda_device_id{d}, per_device_memory.back().get());
    }
    is_initialized = true;
}

#endif

}
}