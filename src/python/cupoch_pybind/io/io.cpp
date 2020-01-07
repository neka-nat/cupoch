#include "cupoch_pybind/io/io.h"

void pybind_io(py::module &m) {
    py::module m_io = m.def_submodule("io");
    pybind_class_io(m_io);
}