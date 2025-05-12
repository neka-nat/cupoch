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
#include "cupoch/visualization/shader/geometry_renderer.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"

using namespace cupoch;
using namespace cupoch::visualization;
using namespace cupoch::visualization::glsl;

bool PointCloudRenderer::Render(const RenderOption &option,
                                const ViewControl &view) {
    if (is_visible_ == false || geometry_ptr_->IsEmpty()) return true;
    const auto &pointcloud = (const geometry::PointCloud &)(*geometry_ptr_);
    bool success = true;
    if (pointcloud.HasNormals()) {
        if (option.point_color_option_ ==
            RenderOption::PointColorOption::Normal) {
            success &= normal_point_shader_.Render(pointcloud, option, view);
        } else {
            success &= phong_point_shader_.Render(pointcloud, option, view);
        }
        if (option.point_show_normal_) {
            success &=
                    simplewhite_normal_shader_.Render(pointcloud, option, view);
        }
    } else {
        success &= simple_point_shader_.Render(pointcloud, option, view);
    }
    return success;
}

bool PointCloudRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloud) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool PointCloudRenderer::UpdateGeometry() {
    simple_point_shader_.InvalidateGeometry();
    phong_point_shader_.InvalidateGeometry();
    normal_point_shader_.InvalidateGeometry();
    simplewhite_normal_shader_.InvalidateGeometry();
    return true;
}

bool LineSetRenderer::Render(const RenderOption &option,
                             const ViewControl &view) {
    if (is_visible_ == false || geometry_ptr_->IsEmpty()) return true;
    return simple_lineset_shader_.Render(*geometry_ptr_, option, view);
}

bool LineSetRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::LineSet) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool LineSetRenderer::UpdateGeometry() {
    simple_lineset_shader_.InvalidateGeometry();
    return true;
}

template <int Dim>
bool GraphRenderer<Dim>::Render(const RenderOption &option,
                                const ViewControl &view) {
    if (is_visible_ == false || geometry_ptr_->IsEmpty()) return true;
    return simple_graph_node_shader_.Render(*geometry_ptr_, option, view) &&
           simple_graph_edge_shader_.Render(*geometry_ptr_, option, view);
}

template <int Dim>
bool GraphRenderer<Dim>::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::Graph) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

template <int Dim>
bool GraphRenderer<Dim>::UpdateGeometry() {
    simple_graph_node_shader_.InvalidateGeometry();
    simple_graph_edge_shader_.InvalidateGeometry();
    return true;
}

template class GraphRenderer<2>;
template class GraphRenderer<3>;

bool TriangleMeshRenderer::Render(const RenderOption &option,
                                  const ViewControl &view) {
    if (is_visible_ == false || geometry_ptr_->IsEmpty()) return true;
    const auto &mesh = (const geometry::TriangleMesh &)(*geometry_ptr_);
    bool success = true;
    if (mesh.HasTriangleNormals() && mesh.HasVertexNormals()) {
        if (option.mesh_color_option_ ==
            RenderOption::MeshColorOption::Normal) {
            success &= normal_mesh_shader_.Render(mesh, option, view);
        } else if (option.mesh_color_option_ ==
                           RenderOption::MeshColorOption::Color &&
                   mesh.HasTriangleUvs() && mesh.HasTexture()) {
            success &= texture_phong_mesh_shader_.Render(mesh, option, view);
        } else {
            success &= phong_mesh_shader_.Render(mesh, option, view);
        }
    } else {  // if normals are not ready
        if (option.mesh_color_option_ == RenderOption::MeshColorOption::Color &&
            mesh.HasTriangleUvs() && mesh.HasTexture()) {
            success &= texture_simple_mesh_shader_.Render(mesh, option, view);
        } else {
            success &= simple_mesh_shader_.Render(mesh, option, view);
        }
    }
    if (option.mesh_show_wireframe_) {
        success &= simplewhite_wireframe_shader_.Render(mesh, option, view);
    }
    return success;
}

bool TriangleMeshRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool TriangleMeshRenderer::UpdateGeometry() {
    simple_mesh_shader_.InvalidateGeometry();
    texture_simple_mesh_shader_.InvalidateGeometry();
    phong_mesh_shader_.InvalidateGeometry();
    texture_phong_mesh_shader_.InvalidateGeometry();
    normal_mesh_shader_.InvalidateGeometry();
    simplewhite_wireframe_shader_.InvalidateGeometry();
    return true;
}

bool VoxelGridRenderer::Render(const RenderOption &option,
                               const ViewControl &view) {
    if (is_visible_ == false || geometry_ptr_->IsEmpty()) return true;
    if (option.mesh_show_wireframe_) {
        return simple_shader_for_voxel_grid_line_.Render(*geometry_ptr_, option,
                                                         view);
    } else {
        return phong_shader_for_voxel_grid_face_.Render(*geometry_ptr_, option,
                                                        view);
    }
}

bool VoxelGridRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::VoxelGrid) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool VoxelGridRenderer::UpdateGeometry() {
    simple_shader_for_voxel_grid_line_.InvalidateGeometry();
    phong_shader_for_voxel_grid_face_.InvalidateGeometry();
    return true;
}

bool OccupancyGridRenderer::Render(const RenderOption &option,
                                   const ViewControl &view) {
    if (is_visible_ == false || geometry_ptr_->IsEmpty()) return true;
    return phong_shader_for_occupancy_grid_.Render(*geometry_ptr_, option,
                                                   view);
}

bool OccupancyGridRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::OccupancyGrid) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool OccupancyGridRenderer::UpdateGeometry() {
    phong_shader_for_occupancy_grid_.InvalidateGeometry();
    return true;
}

bool DistanceTransformRenderer::Render(const RenderOption &option,
                                       const ViewControl &view) {
    if (is_visible_ == false || geometry_ptr_->IsEmpty()) return true;
    return simple_shader_for_distance_transform_.Render(*geometry_ptr_, option,
                                                        view);
}

bool DistanceTransformRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::DistanceTransform) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool DistanceTransformRenderer::UpdateGeometry() {
    simple_shader_for_distance_transform_.InvalidateGeometry();
    return true;
}

bool ImageRenderer::Render(const RenderOption &option,
                           const ViewControl &view) {
    if (is_visible_ == false || geometry_ptr_->IsEmpty()) return true;
    return image_shader_.Render(*geometry_ptr_, option, view);
}

bool ImageRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::Image) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool ImageRenderer::UpdateGeometry() {
    image_shader_.InvalidateGeometry();
    return true;
}

bool CoordinateFrameRenderer::Render(const RenderOption &option,
                                     const ViewControl &view) {
    if (is_visible_ == false || geometry_ptr_->IsEmpty()) return true;
    if (option.show_coordinate_frame_ == false) return true;
    const auto &mesh = (const geometry::TriangleMesh &)(*geometry_ptr_);
    return phong_shader_.Render(mesh, option, view);
}

bool CoordinateFrameRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::TriangleMesh) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool CoordinateFrameRenderer::UpdateGeometry() {
    phong_shader_.InvalidateGeometry();
    return true;
}

bool GridLineRenderer::Render(const RenderOption &option,
                              const ViewControl &view) {
    if (is_visible_ == false || geometry_ptr_->IsEmpty()) return true;
    if (option.show_grid_line_ == false) return true;
    return simple_grid_line_shader_.Render(*geometry_ptr_, option, view);
}

bool GridLineRenderer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    if (geometry_ptr->GetGeometryType() !=
        geometry::Geometry::GeometryType::LineSet) {
        return false;
    }
    geometry_ptr_ = geometry_ptr;
    return UpdateGeometry();
}

bool GridLineRenderer::UpdateGeometry() {
    simple_grid_line_shader_.InvalidateGeometry();
    return true;
}
