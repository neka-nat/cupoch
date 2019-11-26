#include "cupoc_pybind/cupoc_pybind.h"
#include "cupoc_pybind/geometry/geometry.h"
#include "cupoc_pybind/registration/registration.h"

PYBIND11_MODULE(cupoc, m) {
    m.doc() = "CUDA-based point cloud library";

    pybind_geometry(m);
    pybind_registration(m);
}