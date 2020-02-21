#include "cupoch/visualization/shader/texture_simple_shader.h"

#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/visualization/shader/shader.h"
#include "cupoch/visualization/utility/color_map.h"
#include "cupoch/utility/console.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace cupoch;
using namespace cupoch::visualization;
using namespace cupoch::visualization::glsl;

namespace {

struct copy_trianglemesh_functor {
    copy_trianglemesh_functor(const Eigen::Vector3f* vertices, const int* triangles,
                              const Eigen::Vector2f* triangle_uvs)
                              : vertices_(vertices), triangles_(triangles), triangle_uvs_(triangle_uvs) {};
    const Eigen::Vector3f* vertices_;
    const int* triangles_;
    const Eigen::Vector2f* triangle_uvs_;
    __device__
    thrust::tuple<Eigen::Vector3f, Eigen::Vector2f> operator() (size_t k) const {
        int vi = triangles_[k];
        return thrust::make_tuple(vertices_[vi], triangle_uvs_[k]);
    }
};

}

bool TextureSimpleShader::Compile() {
    if (CompileShaders(texture_simple_vertex_shader, NULL,
                       texture_simple_fragment_shader) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_uv_ = glGetAttribLocation(program_, "vertex_uv");
    texture_ = glGetUniformLocation(program_, "diffuse_texture");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void TextureSimpleShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool TextureSimpleShader::BindGeometry(const geometry::Geometry &geometry,
                                       const RenderOption &option,
                                       const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace InvalidateGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Prepare data to be passed to GPU
    const size_t num_data_size = GetDataSize(geometry);

    // Create buffers and bind the geometry
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, num_data_size * sizeof(Eigen::Vector3f), 0, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[0], vertex_position_buffer_, cudaGraphicsMapFlagsNone));
    glGenBuffers(1, &vertex_uv_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffer_);
    glBufferData(GL_ARRAY_BUFFER, num_data_size * sizeof(Eigen::Vector2f), 0, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[1], vertex_uv_buffer_, cudaGraphicsMapFlagsNone));

    Eigen::Vector3f* raw_points_ptr;
    Eigen::Vector2f* raw_uvs_ptr;
    size_t n_bytes;
    cudaSafeCall(cudaGraphicsMapResources(2, cuda_graphics_resources_));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_points_ptr, &n_bytes, cuda_graphics_resources_[0]));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_uvs_ptr, &n_bytes, cuda_graphics_resources_[1]));
    thrust::device_ptr<Eigen::Vector3f> dev_points_ptr = thrust::device_pointer_cast(raw_points_ptr);
    thrust::device_ptr<Eigen::Vector2f> dev_uvs_ptr = thrust::device_pointer_cast(raw_uvs_ptr);

    if (PrepareBinding(geometry, option, view, dev_points_ptr, dev_uvs_ptr) == false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }
    Unmap(2);
    bound_ = true;
    return true;
}

bool TextureSimpleShader::RenderGeometry(const geometry::Geometry &geometry,
                                         const RenderOption &option,
                                         const ViewControl &view) {
    if (PrepareRendering(geometry, option, view) == false) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());

    glUniform1i(texture_, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_buffer_);

    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(vertex_uv_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffer_);
    glVertexAttribPointer(vertex_uv_, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_uv_);
    return true;
}

void TextureSimpleShader::UnbindGeometry() {
    if (bound_) {
        cudaSafeCall(cudaGraphicsUnregisterResource(cuda_graphics_resources_[0]));
        cudaSafeCall(cudaGraphicsUnregisterResource(cuda_graphics_resources_[1]));
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_uv_buffer_);
        glDeleteTextures(1, &texture_buffer_);
        bound_ = false;
    }
}

bool TextureSimpleShaderForTriangleMesh::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::TriangleMesh) {
        PrintShaderWarning("Rendering type is not geometry::TriangleMesh.");
        return false;
    }
    if (option.mesh_show_back_face_) {
        glDisable(GL_CULL_FACE);
    } else {
        glEnable(GL_CULL_FACE);
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    if (option.mesh_show_wireframe_) {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0, 1.0);
    } else {
        glDisable(GL_POLYGON_OFFSET_FILL);
    }
    return true;
}

bool TextureSimpleShaderForTriangleMesh::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector2f> &uvs) {
    if (geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::TriangleMesh) {
        PrintShaderWarning("Rendering type is not geometry::TriangleMesh.");
        return false;
    }
    const geometry::TriangleMesh &mesh =
            (const geometry::TriangleMesh &)geometry;
    if (mesh.HasTriangles() == false) {
        PrintShaderWarning("Binding failed with empty triangle mesh.");
        return false;
    }
    copy_trianglemesh_functor func(thrust::raw_pointer_cast(mesh.vertices_.data()),
                                   (int*)(thrust::raw_pointer_cast(mesh.triangles_.data())),
                                   thrust::raw_pointer_cast(mesh.triangle_uvs_.data()));
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(mesh.triangles_.size() * 3),
                      make_tuple_iterator(points, uvs), func);

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_buffer_);

    GLenum format;
    switch (mesh.texture_.num_of_channels_) {
        case 1: {
            format = GL_RED;
            break;
        }
        case 3: {
            format = GL_RGB;
            break;
        }
        case 4: {
            format = GL_RGBA;
            break;
        }
        default: {
            utility::LogWarning("Unknown format, abort!");
            return false;
        }
    }

    GLenum type;
    switch (mesh.texture_.bytes_per_channel_) {
        case 1: {
            type = GL_UNSIGNED_BYTE;
            break;
        }
        case 2: {
            type = GL_UNSIGNED_SHORT;
            break;
        }
        case 4: {
            type = GL_FLOAT;
            break;
        }
        default: {
            utility::LogWarning("Unknown format, abort!");
            return false;
        }
    }
    glTexImage2D(GL_TEXTURE_2D, 0, format, mesh.texture_.width_,
                 mesh.texture_.height_, 0, format, type,
                 thrust::raw_pointer_cast(mesh.texture_.data_.data()));

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(mesh.triangles_.size() * 3);
    return true;
}

size_t TextureSimpleShaderForTriangleMesh::GetDataSize(const geometry::Geometry &geometry) const {
    return ((const geometry::TriangleMesh &)geometry).triangles_.size() * 3;
}