#include "cupoch/collision/primitives.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/geometry/voxelgrid.h"

#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/collision/collision.h"

using namespace cupoch;
using namespace cupoch::collision;

void pybind_primitives(py::module &m) {
    py::class_<Sphere, std::shared_ptr<Sphere>>
            sphere(m, "Sphere",
                   "Sphere class. A sphere consists of a center point "
                   "coordinate, and radius.");
    py::detail::bind_default_constructor<Sphere>(sphere);
    py::detail::bind_copy_functions<Sphere>(sphere);
    sphere.def(py::init<float>(),
               "Create a Sphere", "radius"_a)
          .def(py::init<float, const Eigen::Vector3f&>(),
               "Create a Sphere", "radius"_a, "center"_a)
          .def("create_voxel_grid", &Sphere::CreateVoxelGrid)
          .def("create_voxel_grid_with_sweeping", &Sphere::CreateVoxelGridWithSweeping)
          .def("create_triangle_mesh", &Sphere::CreateTriangleMesh);
}