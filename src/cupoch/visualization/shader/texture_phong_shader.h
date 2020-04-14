#pragma once

#include <Eigen/Core>
#include <thrust/device_ptr.h>

#include "cupoch/visualization/shader/shader_wrapper.h"

namespace cupoch {
namespace visualization {

namespace glsl {

class TexturePhongShader : public ShaderWrapper {
public:
    ~TexturePhongShader() override { Release(); }

protected:
    TexturePhongShader(const std::string &name) : ShaderWrapper(name) {
        Compile();
    }

protected:
    bool Compile() final;
    void Release() final;
    bool BindGeometry(const geometry::Geometry &geometry,
                      const RenderOption &option,
                      const ViewControl &view) final;
    bool RenderGeometry(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view) final;
    void UnbindGeometry(bool finalize = false) final;

protected:
    virtual bool PrepareRendering(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) = 0;
    virtual bool PrepareBinding(const geometry::Geometry &geometry,
                                const RenderOption &option,
                                const ViewControl &view,
                                thrust::device_ptr<Eigen::Vector3f> &points,
                                thrust::device_ptr<Eigen::Vector3f> &normals,
                                thrust::device_ptr<Eigen::Vector2f> &uvs,
                                thrust::device_ptr<uint8_t> &texture_image) = 0;
    virtual size_t GetDataSize(const geometry::Geometry &geometry) const = 0;
    virtual size_t GetTextureSize(const geometry::Geometry &geometry) const = 0;
    virtual size_t GetTextureHeight(const geometry::Geometry &geometry) const = 0;
    virtual size_t GetTextureWidth(const geometry::Geometry &geometry) const = 0;
protected:
    void SetLighting(const ViewControl &view, const RenderOption &option);

protected:
    GLuint vertex_position_;
    GLuint vertex_position_buffer_;
    GLuint vertex_uv_;
    GLuint vertex_uv_buffer_;
    GLuint vertex_normal_;
    GLuint vertex_normal_buffer_;
    GLuint MVP_;
    GLuint V_;
    GLuint M_;
    GLuint light_position_world_;
    GLuint light_color_;
    GLuint light_diffuse_power_;
    GLuint light_specular_power_;
    GLuint light_specular_shininess_;
    GLuint light_ambient_;

    GLuint diffuse_texture_;
    GLuint diffuse_texture_buffer_;
    GLuint diffuse_texture_pixel_buffer_;

    // At most support 4 lights
    gl_helper::GLMatrix4f light_position_world_data_;
    gl_helper::GLMatrix4f light_color_data_;
    gl_helper::GLVector4f light_diffuse_power_data_;
    gl_helper::GLVector4f light_specular_power_data_;
    gl_helper::GLVector4f light_specular_shininess_data_;
    gl_helper::GLVector4f light_ambient_data_;
};

class TexturePhongShaderForTriangleMesh : public TexturePhongShader {
public:
    TexturePhongShaderForTriangleMesh()
        : TexturePhongShader("TexturePhongShaderForTriangleMesh") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points,
                        thrust::device_ptr<Eigen::Vector3f> &normals,
                        thrust::device_ptr<Eigen::Vector2f> &uvs,
                        thrust::device_ptr<uint8_t> &texture_image) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
    size_t GetTextureSize(const geometry::Geometry &geometry) const final;
    size_t GetTextureHeight(const geometry::Geometry &geometry) const final;
    size_t GetTextureWidth(const geometry::Geometry &geometry) const final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace cupoch