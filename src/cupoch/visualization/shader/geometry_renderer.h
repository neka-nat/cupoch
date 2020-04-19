#pragma once

#include "cupoch/geometry/geometry.h"
#include "cupoch/visualization/shader/simple_shader.h"
#include "cupoch/visualization/shader/phong_shader.h"
#include "cupoch/visualization/shader/normal_shader.h"
#include "cupoch/visualization/shader/simple_white_shader.h"
#include "cupoch/visualization/shader/image_shader.h"
#include "cupoch/visualization/shader/texture_simple_shader.h"
#include "cupoch/visualization/shader/texture_phong_shader.h"

namespace cupoch {
namespace visualization {

namespace glsl {

class GeometryRenderer {
public:
    virtual ~GeometryRenderer() {}

public:
    virtual bool Render(const RenderOption &option,
                        const ViewControl &view) = 0;

    /// Function to add geometry to the renderer
    /// 1. After calling the function, the renderer owns the geometry object.
    /// 2. This function returns FALSE if the geometry type is not matched to
    /// the renderer.
    /// 3. If an added geometry is changed, programmer must call
    /// UpdateGeometry() to notify the renderer.
    virtual bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) = 0;

    /// Function to update geometry
    /// Programmer must call this function to notify a change of the geometry
    virtual bool UpdateGeometry() = 0;

    bool HasGeometry() const { return bool(geometry_ptr_); }
    std::shared_ptr<const geometry::Geometry> GetGeometry() const {
        return geometry_ptr_;
    }

    bool HasGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) const {
        return geometry_ptr_ == geometry_ptr;
    }

    bool IsVisible() const { return is_visible_; }
    void SetVisible(bool visible) { is_visible_ = visible; };

protected:
    std::shared_ptr<const geometry::Geometry> geometry_ptr_;
    bool is_visible_ = true;
};

class PointCloudRenderer : public GeometryRenderer {
public:
    ~PointCloudRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForPointCloud simple_point_shader_;
    PhongShaderForPointCloud phong_point_shader_;
    NormalShaderForPointCloud normal_point_shader_;
    SimpleWhiteShaderForPointCloudNormal simplewhite_normal_shader_;
};

class TriangleMeshRenderer : public GeometryRenderer {
public:
    ~TriangleMeshRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForTriangleMesh simple_mesh_shader_;
    TextureSimpleShaderForTriangleMesh texture_simple_mesh_shader_;
    PhongShaderForTriangleMesh phong_mesh_shader_;
    TexturePhongShaderForTriangleMesh texture_phong_mesh_shader_;
    NormalShaderForTriangleMesh normal_mesh_shader_;
    SimpleWhiteShaderForTriangleMeshWireFrame simplewhite_wireframe_shader_;
};

class VoxelGridRenderer : public GeometryRenderer {
public:
    ~VoxelGridRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    SimpleShaderForVoxelGridLine simple_shader_for_voxel_grid_line_;
    PhongShaderForVoxelGridFace phong_shader_for_voxel_grid_face_;
};

class OccupancyGridRenderer : public GeometryRenderer {
public:
    ~OccupancyGridRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    PhongShaderForOccupancyGrid phong_shader_for_occupancy_grid_;
};

class CoordinateFrameRenderer : public GeometryRenderer {
public:
    ~CoordinateFrameRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    PhongShaderForTriangleMesh phong_shader_;
};

class ImageRenderer : public GeometryRenderer {
public:
    ~ImageRenderer() override {}

public:
    bool Render(const RenderOption &option, const ViewControl &view) override;
    bool AddGeometry(
            std::shared_ptr<const geometry::Geometry> geometry_ptr) override;
    bool UpdateGeometry() override;

protected:
    ImageShaderForImage image_shader_;
};

}  // namespace glsl
}  // namespace visualization
}  // namespace cupoch