#pragma once

#include "cupoch/geometry/image.h"
#include "cupoch/visualization/shader/shader_wrapper.h"

namespace cupoch {
namespace visualization {

namespace glsl {

class ImageShader : public ShaderWrapper {
public:
    ~ImageShader() override { Release(); }

protected:
    ImageShader(const std::string &name) : ShaderWrapper(name) { Compile(); }

protected:
    bool Compile() final;
    void Release() final;
    bool BindGeometry(const geometry::Geometry &geometry,
                      const RenderOption &option,
                      const ViewControl &view) final;
    bool RenderGeometry(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view) final;
    void UnbindGeometry() final;

protected:
    virtual bool PrepareRendering(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) = 0;
    virtual bool PrepareBinding(const geometry::Geometry &geometry,
                                const RenderOption &option,
                                const ViewControl &view,
                                geometry::Image &image) = 0;

protected:
    GLuint vertex_position_;
    GLuint vertex_position_buffer_;
    GLuint vertex_UV_;
    GLuint vertex_UV_buffer_;
    GLuint image_texture_;
    GLuint image_texture_buffer_;
    GLuint vertex_scale_;

    gl_helper::GLVector3f vertex_scale_data_;
};

class ImageShaderForImage : public ImageShader {
public:
    ImageShaderForImage() : ImageShader("ImageShaderForImage") {}

protected:
    virtual bool PrepareRendering(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) final;
    virtual bool PrepareBinding(const geometry::Geometry &geometry,
                                const RenderOption &option,
                                const ViewControl &view,
                                geometry::Image &render_image) final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace cupoch