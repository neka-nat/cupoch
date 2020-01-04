#pragma once

#include "cupoch_pybind/cupoch_pybind.h"

void pybind_visualization(py::module &m);

void pybind_renderoption(py::module &m);
void pybind_viewcontrol(py::module &m);
void pybind_visualizer(py::module &m);
void pybind_visualization_utility(py::module &m);

void pybind_renderoption_method(py::module &m);
void pybind_viewcontrol_method(py::module &m);
void pybind_visualizer_method(py::module &m);
void pybind_visualization_utility_methods(py::module &m);