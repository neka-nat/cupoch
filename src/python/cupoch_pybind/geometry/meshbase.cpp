#include "cupoch/geometry/meshbase.h"
#include "cupoch/geometry/pointcloud.h"

#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_meshbase(py::module &m) {
    py::class_<geometry::MeshBase, PyGeometry3D<geometry::MeshBase>,
               std::shared_ptr<geometry::MeshBase>, geometry::Geometry3D>
            meshbase(m, "MeshBase",
                     "MeshBase class. Triangle mesh contains vertices. "
                     "Optionally, the mesh "
                     "may also contain vertex normals and vertex colors.");
    py::detail::bind_default_constructor<geometry::MeshBase>(meshbase);
    py::detail::bind_copy_functions<geometry::MeshBase>(meshbase);
    py::enum_<geometry::MeshBase::SimplificationContraction>(
            m, "SimplificationContraction")
            .value("Average",
                   geometry::MeshBase::SimplificationContraction::Average,
                   "The vertex positions are computed by the averaging.")
            .value("Quadric",
                   geometry::MeshBase::SimplificationContraction::Quadric,
                   "The vertex positions are computed by minimizing the "
                   "distance to the adjacent triangle planes.")
            .export_values();
    py::enum_<geometry::MeshBase::FilterScope>(m, "FilterScope")
            .value("All", geometry::MeshBase::FilterScope::All,
                   "All properties (color, normal, vertex position) are "
                   "filtered.")
            .value("Color", geometry::MeshBase::FilterScope::Color,
                   "Only the color values are filtered.")
            .value("Normal", geometry::MeshBase::FilterScope::Normal,
                   "Only the normal values are filtered.")
            .value("Vertex", geometry::MeshBase::FilterScope::Vertex,
                   "Only the vertex positions are filtered.")
            .export_values();
    meshbase.def("__repr__",
                 [](const geometry::MeshBase &mesh) {
                     return std::string("geometry::MeshBase with ") +
                            std::to_string(mesh.vertices_.size()) + " points";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_vertices", &geometry::MeshBase::HasVertices,
                 "Returns ``True`` if the mesh contains vertices.")
            .def("has_vertex_normals", &geometry::MeshBase::HasVertexNormals,
                 "Returns ``True`` if the mesh contains vertex normals.")
            .def("has_vertex_colors", &geometry::MeshBase::HasVertexColors,
                 "Returns ``True`` if the mesh contains vertex colors.")
            .def("normalize_normals", &geometry::MeshBase::NormalizeNormals,
                 "Normalize vertex normals to legnth 1.")
            .def("paint_uniform_color", &geometry::MeshBase::PaintUniformColor,
                 "Assigns each vertex in the MeshBase the same color.")
            .def_property("vertices", [] (geometry::MeshBase &mesh) {return wrapper::device_vector_vector3f(mesh.vertices_);},
                                      [] (geometry::MeshBase &mesh, const wrapper::device_vector_vector3f& vec) {wrapper::FromWrapper(mesh.vertices_, vec);})
            .def_property("vertex_normals", [] (geometry::MeshBase &mesh) {return wrapper::device_vector_vector3f(mesh.vertex_normals_);},
                                            [] (geometry::MeshBase &mesh, const wrapper::device_vector_vector3f& vec) {wrapper::FromWrapper(mesh.vertex_normals_, vec);})
            .def_property("vertex_colors", [] (geometry::MeshBase &mesh) {return wrapper::device_vector_vector3f(mesh.vertex_colors_);},
                                           [] (geometry::MeshBase &mesh, const wrapper::device_vector_vector3f& vec) {wrapper::FromWrapper(mesh.vertex_colors_, vec);});
    docstring::ClassMethodDocInject(
            m, "MeshBase", "has_vertex_normals",
            {{"normalized",
              "Set to ``True`` to normalize the normal to length 1."}});
    docstring::ClassMethodDocInject(m, "MeshBase", "has_vertices");
    docstring::ClassMethodDocInject(m, "MeshBase", "normalize_normals");
    docstring::ClassMethodDocInject(
            m, "MeshBase", "paint_uniform_color",
            {{"color", "RGB color for the PointCloud."}});
}

void pybind_meshbase_methods(py::module &m) {}