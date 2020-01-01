#pragma once

#include <Eigen/Core>
#include <thrust/device_ptr.h>

#include "cupoch/visualization/shader/shader_wrapper.h"

namespace cupoch {
namespace visualization {

namespace glsl {

class SimpleBlackShader : public ShaderWrapper {
public:
    ~SimpleBlackShader() override { Release(); }

protected:
    SimpleBlackShader(const std::string &name) : ShaderWrapper(name) {
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
    void UnbindGeometry() final;

protected:
    virtual bool PrepareRendering(const geometry::Geometry &geometry,
                                  const RenderOption &option,
                                  const ViewControl &view) = 0;
    virtual bool PrepareBinding(const geometry::Geometry &geometry,
                                const RenderOption &option,
                                const ViewControl &view,
                                thrust::device_ptr<Eigen::Vector3f> &points) = 0;
    virtual size_t GetDataSize(const geometry::Geometry &geometry) const = 0;
protected:
    GLuint vertex_position_;
    GLuint vertex_position_buffer_;
    GLuint MVP_;
};

class SimpleBlackShaderForPointCloudNormal : public SimpleBlackShader {
public:
    SimpleBlackShaderForPointCloudNormal()
        : SimpleBlackShader("SimpleBlackShaderForPointCloudNormal") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

class SimpleBlackShaderForTriangleMeshWireFrame : public SimpleBlackShader {
public:
    SimpleBlackShaderForTriangleMeshWireFrame()
        : SimpleBlackShader("SimpleBlackShaderForTriangleMeshWireFrame") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace cupoch