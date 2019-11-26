#include "cupoc_pybind/geometry/geometry.h"

using namespace cupoc;

void pybind_geometry(py::module &m) {
    py::module m_submodule = m.def_submodule("geometry");
    pybind_kdtreeflann(m_submodule);
    pybind_pointcloud(m_submodule);
}