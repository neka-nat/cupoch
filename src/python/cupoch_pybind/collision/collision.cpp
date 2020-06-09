#include "cupoch_pybind/collision/collision.h"
#include "cupoch_pybind/docstring.h"

#include "cupoch/collision/collision.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/geometry/lineset.h"
#include "cupoch/geometry/occupancygrid.h"

using namespace cupoch;

void pybind_collision_methods(py::module &m) {
    py::class_<collision::CollisionResult, std::shared_ptr<collision::CollisionResult>>
            col_res(m, "CollitionResult",
                    "Collision result class.");
    py::detail::bind_default_constructor<collision::CollisionResult>(col_res);
    py::detail::bind_copy_functions<collision::CollisionResult>(col_res);
    col_res.def("is_collided", &collision::CollisionResult::IsCollided)
           .def_property("collision_index_pairs", [] (collision::CollisionResult &res) {return wrapper::device_vector_vector2i(res.collision_index_pairs_);},
                         [] (collision::CollisionResult &res, const wrapper::device_vector_vector2i& vec) {wrapper::FromWrapper(res.collision_index_pairs_, vec);});
    py::enum_<collision::CollisionResult::CollisionType> collision_type(col_res, "Type",
                                                                        py::arithmetic());
    collision_type
            .value("Unspecified", collision::CollisionResult::CollisionType::Unspecified)
            .value("Primitives", collision::CollisionResult::CollisionType::Primitives)
            .value("VoxelGrid", collision::CollisionResult::CollisionType::VoxelGrid)
            .value("OccupancyGrid", collision::CollisionResult::CollisionType::OccupancyGrid)
            .value("LineSet", collision::CollisionResult::CollisionType::LineSet)
            .export_values();

    m.def("compute_intersection", py::overload_cast<const geometry::VoxelGrid&, const geometry::VoxelGrid&, float>(&collision::ComputeIntersection));
    m.def("compute_intersection", py::overload_cast<const geometry::VoxelGrid&, const geometry::LineSet&, float>(&collision::ComputeIntersection));
    m.def("compute_intersection", py::overload_cast<const geometry::LineSet&, const geometry::VoxelGrid&, float>(&collision::ComputeIntersection));
    m.def("compute_intersection", py::overload_cast<const geometry::VoxelGrid&, const geometry::OccupancyGrid&, float>(&collision::ComputeIntersection));
    m.def("compute_intersection", py::overload_cast<const geometry::OccupancyGrid&, const geometry::VoxelGrid&, float>(&collision::ComputeIntersection));
}

void pybind_collision(py::module &m) {
    py::module m_submodule = m.def_submodule("collision");
    pybind_collision_methods(m_submodule);
    pybind_primitives(m_submodule);
}