#include "cupoch_pybind/collision/collision.h"
#include "cupoch_pybind/docstring.h"

#include "cupoch/collision/collision.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/geometry/occupancygrid.h"

using namespace cupoch;

void pybind_collision_methods(py::module &m) {
    m.def("compute_intersection", py::overload_cast<const geometry::VoxelGrid&, const geometry::VoxelGrid&>(&collision::ComputeIntersection));
    m.def("compute_intersection", py::overload_cast<const geometry::VoxelGrid&, const geometry::OccupancyGrid&>(&collision::ComputeIntersection));
    m.def("compute_intersection", py::overload_cast<const geometry::OccupancyGrid&, const geometry::VoxelGrid&>(&collision::ComputeIntersection));
}

void pybind_collision(py::module &m) {
    py::module m_submodule = m.def_submodule("collision");
    pybind_collision_methods(m_submodule);
    pybind_primitives(m_submodule);
}