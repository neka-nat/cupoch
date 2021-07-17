/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
**/
#include "cupoch_pybind/collision/collision.h"

#include "cupoch/collision/collision.h"
#include "cupoch/geometry/lineset.h"
#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch_pybind/docstring.h"

using namespace cupoch;

void pybind_collision_methods(py::module& m) {
    py::class_<collision::CollisionResult,
               std::shared_ptr<collision::CollisionResult>>
            col_res(m, "CollitionResult", "Collision result class.");
    py::detail::bind_default_constructor<collision::CollisionResult>(col_res);
    py::detail::bind_copy_functions<collision::CollisionResult>(col_res);
    col_res.def("is_collided", &collision::CollisionResult::IsCollided)
            .def("__repr__",
                    [](const collision::CollisionResult &res) {
                        return std::string("collision::CollisionResult with ") +
                               std::to_string(res.collision_index_pairs_.size()) + " collisions.";
                    })
            .def("get_first_collision_indices",
                    [] (const collision::CollisionResult& self) {
                        return wrapper::device_vector_size_t(self.GetFirstCollisionIndices());
                    })
            .def("get_second_collision_indices",
                    [] (const collision::CollisionResult& self) {
                        return wrapper::device_vector_size_t(self.GetSecondCollisionIndices());
                    })
            .def_readwrite("first", &collision::CollisionResult::first_)
            .def_readwrite("second", &collision::CollisionResult::second_)
            .def_property(
                    "collision_index_pairs",
                    [](collision::CollisionResult& res) {
                        return wrapper::device_vector_vector2i(
                                res.collision_index_pairs_);
                    },
                    [](collision::CollisionResult& res,
                       const wrapper::device_vector_vector2i& vec) {
                        wrapper::FromWrapper(res.collision_index_pairs_, vec);
                    });
    py::enum_<collision::CollisionResult::CollisionType> collision_type(
            col_res, "Type", py::arithmetic());
    collision_type
            .value("Unspecified",
                   collision::CollisionResult::CollisionType::Unspecified)
            .value("Primitives",
                   collision::CollisionResult::CollisionType::Primitives)
            .value("VoxelGrid",
                   collision::CollisionResult::CollisionType::VoxelGrid)
            .value("OccupancyGrid",
                   collision::CollisionResult::CollisionType::OccupancyGrid)
            .value("LineSet",
                   collision::CollisionResult::CollisionType::LineSet)
            .export_values();

    m.def("compute_intersection",
          py::overload_cast<const geometry::VoxelGrid&,
                            const geometry::VoxelGrid&, float>(
                  &collision::ComputeIntersection),
          "Compute intersection betweeen VoxelGrid and VoxelGrid.",
          "voxelgrid1"_a, "voxelgrid2"_a, "margin"_a = 0.0f);
    m.def("compute_intersection",
          py::overload_cast<const geometry::VoxelGrid&,
                            const geometry::LineSet<3>&, float>(
                  &collision::ComputeIntersection),
          "Compute intersection betweeen VoxelGrid and LineSet.",
          "voxelgrid"_a, "lineset"_a, "margin"_a = 0.0f);
    m.def("compute_intersection",
          py::overload_cast<const geometry::LineSet<3>&,
                            const geometry::VoxelGrid&, float>(
                  &collision::ComputeIntersection),
          "Compute intersection betweeen LineSet and VoxelGrid.",
          "lineset"_a, "voxelgrid"_a, "margin"_a = 0.0f);
    m.def("compute_intersection",
          py::overload_cast<const geometry::VoxelGrid&,
                            const geometry::OccupancyGrid&, float>(
                  &collision::ComputeIntersection),
          "voxelgrid"_a, "lineset"_a, "margin"_a = 0.0f);
    m.def("compute_intersection",
          py::overload_cast<const geometry::OccupancyGrid&,
                            const geometry::VoxelGrid&, float>(
                  &collision::ComputeIntersection),
          "Compute intersection betweeen OccupancyGrid and VoxelGrid.",
          "occgrid"_a, "voxelgrid"_a, "margin"_a = 0.0f);
    m.def("compute_intersection",
          py::overload_cast<const geometry::LineSet<3>&,
                            const geometry::OccupancyGrid&, float>(
                  &collision::ComputeIntersection),
          "lineset"_a, "occgrid"_a, "margin"_a = 0.0f);
    m.def("compute_intersection",
          py::overload_cast<const geometry::OccupancyGrid&,
                            const geometry::LineSet<3>&, float>(
                  &collision::ComputeIntersection),
          "occgrid"_a, "linset"_a, "margin"_a = 0.0f);
    m.def("compute_intersection",
          [](const wrapper::device_vector_primitives& primitives,
             const geometry::VoxelGrid& voxel, float margin) {
              return collision::ComputeIntersection(primitives.data_, voxel,
                                                    margin);
          },
          "Compute intersection betweeen Primitives and VoxelGrid.",
          "primitives"_a, "voxelgrid"_a, "margin"_a = 0.0f);
    m.def("compute_intersection",
          [](const geometry::VoxelGrid& voxel,
             const wrapper::device_vector_primitives& primitives,
             float margin) {
              return collision::ComputeIntersection(voxel, primitives.data_,
                                                    margin);
          },
          "Compute intersection betweeen VoxelGrid and Primitives.",
          "voxelgrid"_a, "primitives"_a, "margin"_a = 0.0f);
    m.def("compute_intersection",
          [](const wrapper::device_vector_primitives& primitives,
             const geometry::OccupancyGrid& occgrid, float margin) {
              return collision::ComputeIntersection(primitives.data_, occgrid,
                                                    margin);
          },
          "Compute intersection betweeen Primitives and OccupancyGrid.",
          "primitives"_a, "occgrid"_a, "margin"_a = 0.0f);
    m.def("compute_intersection",
          [](const geometry::OccupancyGrid& occgrid,
             const wrapper::device_vector_primitives& primitives,
             float margin) {
              return collision::ComputeIntersection(occgrid, primitives.data_,
                                                    margin);
          },
          "Compute intersection betweeen OccupancyGrid and Primitives.",
          "occgrid"_a, "primitives"_a, "margin"_a = 0.0f);
}

void pybind_collision(py::module& m) {
    py::module m_submodule = m.def_submodule("collision");
    pybind_collision_methods(m_submodule);
    pybind_primitives(m_submodule);
}