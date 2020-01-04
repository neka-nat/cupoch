#include "cupoch_pybind/visualization/visualization.h"

void pybind_visualization(py::module &m) {
    py::module m_visualization = m.def_submodule("visualization");
    pybind_renderoption(m_visualization);
    pybind_viewcontrol(m_visualization);
    pybind_visualizer(m_visualization);
    pybind_renderoption_method(m_visualization);
    pybind_viewcontrol_method(m_visualization);
    pybind_visualizer_method(m_visualization);
    pybind_visualization_utility_methods(m_visualization);
}