#include "cupoch_pybind/collision/collision.h"
#include "cupoch_pybind/docstring.h"

using namespace cupoch;

void pybind_collision(py::module &m) {
    py::module m_submodule = m.def_submodule("collision");
    pybind_primitives(m_submodule);
}