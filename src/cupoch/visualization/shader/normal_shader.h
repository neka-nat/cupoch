#pragma once

#include <thrust/device_ptr.h>

#include <Eigen/Core>

#include "cupoch/visualization/shader/shader_wrapper.h"

namespace cupoch {
namespace visualization {

namespace glsl {

class NormalShader : public ShaderWrapper {
public:
    ~NormalShader() override { Release(); }

protected:
    NormalShader(const std::string &name) : ShaderWrapper(name) { Compile(); }

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
    virtual bool PrepareBinding(
            const geometry::Geometry &geometry,
            const RenderOption &option,
            const ViewControl &view,
            thrust::device_ptr<Eigen::Vector3f> &points,
            thrust::device_ptr<Eigen::Vector3f> &normals) = 0;
    virtual size_t GetDataSize(const geometry::Geometry &geometry) const = 0;

protected:
    GLuint vertex_position_;
    GLuint vertex_position_buffer_;
    GLuint vertex_normal_;
    GLuint vertex_normal_buffer_;
    GLuint MVP_;
    GLuint V_;
    GLuint M_;
};

class NormalShaderForPointCloud : public NormalShader {
public:
    NormalShaderForPointCloud() : NormalShader("NormalShaderForPointCloud") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points,
                        thrust::device_ptr<Eigen::Vector3f> &normals) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

class NormalShaderForTriangleMesh : public NormalShader {
public:
    NormalShaderForTriangleMesh()
        : NormalShader("NormalShaderForTriangleMesh") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points,
                        thrust::device_ptr<Eigen::Vector3f> &normals) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace cupoch