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
                    simpleblack_normal_shader_.Render(pointcloud, option, view);
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
    simpleblack_normal_shader_.InvalidateGeometry();
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