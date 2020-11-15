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
#pragma once

#include <thrust/device_ptr.h>

#include <Eigen/Core>

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
    virtual bool PrepareBinding(
            const geometry::Geometry &geometry,
            const RenderOption &option,
            const ViewControl &view,
            thrust::device_ptr<Eigen::Vector3f> &points,
            thrust::device_ptr<Eigen::Vector4f> &colors) = 0;
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
                        thrust::device_ptr<Eigen::Vector4f> &colors) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

class SimpleShaderForLineSet : public SimpleShader {
public:
    SimpleShaderForLineSet() : SimpleShader("SimpleShaderForLineSet") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points,
                        thrust::device_ptr<Eigen::Vector4f> &colors) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

template <int Dim>
class SimpleShaderForGraphNode : public SimpleShader {
public:
    SimpleShaderForGraphNode() : SimpleShader("SimpleShaderForGraphNode") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points,
                        thrust::device_ptr<Eigen::Vector4f> &colors) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

template <int Dim>
class SimpleShaderForGraphEdge : public SimpleShader {
public:
    SimpleShaderForGraphEdge() : SimpleShader("SimpleShaderForGraphEdge") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points,
                        thrust::device_ptr<Eigen::Vector4f> &colors) final;
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
                        thrust::device_ptr<Eigen::Vector4f> &colors) final;
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
                        thrust::device_ptr<Eigen::Vector4f> &colors) final;
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
                        thrust::device_ptr<Eigen::Vector4f> &colors) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

class SimpleShaderForDistanceTransform : public SimpleShader {
public:
    SimpleShaderForDistanceTransform()
        : SimpleShader("SimpleShaderForDistanceTransform") {}

protected:
    bool PrepareRendering(const geometry::Geometry &geometry,
                          const RenderOption &option,
                          const ViewControl &view) final;
    bool PrepareBinding(const geometry::Geometry &geometry,
                        const RenderOption &option,
                        const ViewControl &view,
                        thrust::device_ptr<Eigen::Vector3f> &points,
                        thrust::device_ptr<Eigen::Vector4f> &colors) final;
    size_t GetDataSize(const geometry::Geometry &geometry) const final;
};

}  // namespace glsl

}  // namespace visualization
}  // namespace cupoch