#include "cupoch_pybind/utility/utility.h"

void pybind_utility(py::module &m) {
    py::module m_submodule = m.def_submodule("utility");
    pybind_console(m_submodule);
    pybind_eigen(m_submodule);
}