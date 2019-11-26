#pragma once
#include "cupoc_pybind/cupoc_pybind.h"


void pybind_geometry(py::module &m);

void pybind_pointcloud(py::module &m);
void pybind_kdtreeflann(py::module &m);