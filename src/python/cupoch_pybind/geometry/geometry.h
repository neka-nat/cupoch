#pragma once
#include "cupoch_pybind/cupoch_pybind.h"


void pybind_geometry(py::module &m);

void pybind_pointcloud(py::module &m);
void pybind_kdtreeflann(py::module &m);