#include "cupoc_pybind/cupoc_pybind.h"
#include "cupoc_pybind/geometry/geometry.h"

PYBIND11_MODULE(cupoc, m) {
    m.doc() = "CUDA-based point cloud library";

    pybind_pointcloud(m);
}