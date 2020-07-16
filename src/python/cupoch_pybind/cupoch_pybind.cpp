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
#include "cupoch_pybind/cupoch_pybind.h"

#include "cupoch_pybind/camera/camera.h"
#include "cupoch_pybind/collision/collision.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/integration/integration.h"
#include "cupoch_pybind/io/io.h"
#include "cupoch_pybind/odometry/odometry.h"
#include "cupoch_pybind/registration/registration.h"
#include "cupoch_pybind/utility/utility.h"
#include "cupoch_pybind/visualization/visualization.h"

PYBIND11_MODULE(cupoch, m) {
    m.doc() = "CUDA-based 3D data processing library";

    py::enum_<rmmAllocationMode_t> rmm_mode(m, "AllocationMode",
                                            py::arithmetic());
    rmm_mode.value("CudaDefaultAllocation", CudaDefaultAllocation)
            .value("PoolAllocation", PoolAllocation)
            .value("CudaManagedMemory", CudaManagedMemory)
            .export_values();
    m.def("initialize_allocator", &cupoch::utility::InitializeAllocator,
          py::arg("mode") = CudaDefaultAllocation,
          py::arg("initial_pool_size") = 0, py::arg("logging") = false,
          py::arg("devices") = std::vector<int>());
    cupoch::docstring::FunctionDocInject(
            m, "initialize_allocator",
            {
                    {"mode", "Allocation strategy to use"},
                    {"initial_pool_size",
                     "When pool suballocation is enabled, this is the initial "
                     "pool size in bytes"},
                    {"logging", "Enable logging memory manager events"},
                    {"devices", "List of GPU device IDs to register"},
            });
    pybind_utility(m);
    pybind_camera(m);
    pybind_geometry(m);
    pybind_integration(m);
    pybind_collision(m);
    pybind_io(m);
    pybind_registration(m);
    pybind_odometry(m);
    pybind_visualization(m);
}