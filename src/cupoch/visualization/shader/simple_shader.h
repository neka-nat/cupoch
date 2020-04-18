#pragma once

#include <Eigen/Core>
#include <thrust/device_ptr.h>

#include "cupoch/visualization/shader/shader_wrapper.h"

namespace cupoch {
namespace visualization {

namespace glsl {

class SimpleShader : public ShaderWrapper {
public:
    ~SimpleShader() override { Release(); }

protected:
    SimpleShader(const std::string &name) : ShaderWrapper(name) { Compile(); }

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
                                thrust::device_ptr<Eigen::Vector3f> &colors) = 0;
    virtual size_t GetDataSize(const geometry::Geometry &geometry) const = 0;
protected:
    GLuint vertex_position_;
    GLuint vertex_position_buffer_;
    GLuint vertex_color_;
    GLuint vertex_color_buffer_;
    GLuint MVP_;
};

class SimpleShaderForPointCloud : public SimpleShader {
public:
    SimpleShaderForPointCloud() : SimpleShader("SimpleShaderForPointCloud") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points,
                        thrust::device_ptr<Eigen::Vector3f> &colors) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

class SimpleShaderForAxisAlignedBoundingBox : public SimpleShader {
public:
    SimpleShaderForAxisAlignedBoundingBox()
        : SimpleShader("SimpleShaderForAxisAlignedBoundingBox") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points,
                        thrust::device_ptr<Eigen::Vector3f> &colors) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

class SimpleShaderForTriangleMesh : public SimpleShader {
public:
    SimpleShaderForTriangleMesh()
        : SimpleShader("SimpleShaderForTriangleMesh") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points,
                        thrust::device_ptr<Eigen::Vector3f> &colors) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

class SimpleShaderForVoxelGridLine : public SimpleShader {
public:
    SimpleShaderForVoxelGridLine()
        : SimpleShader("SimpleShaderForVoxelGridLine") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points,
                        thrust::device_ptr<Eigen::Vector3f> &colors) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace cupoch