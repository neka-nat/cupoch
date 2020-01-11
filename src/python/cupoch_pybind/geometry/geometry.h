#pragma once
#include "cupoch_pybind/cupoch_pybind.h"


void pybind_geometry(py::module &m);

void pybind_pointcloud(py::module &m);
void pybind_lineset(py::module &m);
void pybind_meshbase(py::module &m);
void pybind_image(py::module &m);
void pybind_tetramesh(py::module &m);
void pybind_kdtreeflann(py::module &m);
void pybind_boundingvolume(py::module &m);