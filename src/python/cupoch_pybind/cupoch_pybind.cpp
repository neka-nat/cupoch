#include "cupoch_pybind/cupoch_pybind.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/registration/registration.h"

PYBIND11_MODULE(cupoch, m) {
    m.doc() = "CUDA-based 3D data processing library";

    pybind_geometry(m);
    pybind_registration(m);
}