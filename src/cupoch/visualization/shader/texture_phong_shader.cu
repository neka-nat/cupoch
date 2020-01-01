#include "cupoch/visualization/shader/texture_phong_shader.h"

#include "cupoch/geometry/image.h"
#include "cupoch/geometry/pointcloud.h"
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
    copy_trianglemesh_functor(const Eigen::Vector3f* vertices, const Eigen::Vector3f* vertex_normals,
                              const int* triangles, const Eigen::Vector3f* triangle_normals,
                              const Eigen::Vector2f* triangle_uvs,
                              RenderOption::MeshShadeOption shade_option)
                              : vertices_(vertices), vertex_normals_(vertex_normals),
                                triangles_(triangles), triangle_normals_(triangle_normals),
                                triangle_uvs_(triangle_uvs), shade_option_(shade_option) {};
    const Eigen::Vector3f* vertices_;
    const Eigen::Vector3f* vertex_normals_;
    const int* triangles_;
    const Eigen::Vector3f* triangle_normals_;
    const Eigen::Vector2f* triangle_uvs_;
    const RenderOption::MeshShadeOption shade_option_;
    __device__
    thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector2f> operator() (size_t k) const {
        int i = k / 3;
        int vi = triangles_[k];
        if (shade_option_ ==
            RenderOption::MeshShadeOption::FlatShade) {
            return thrust::make_tuple(vertices_[vi], triangle_normals_[i], triangle_uvs_[k]);
        } else {
            return thrust::make_tuple(vertices_[vi], vertex_normals_[vi], triangle_uvs_[k]);
        }
    }
};

}

bool TexturePhongShader::Compile() {
    if (CompileShaders(texture_phong_vertex_shader, NULL,
                       texture_phong_fragment_shader) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_normal_ = glGetAttribLocation(program_, "vertex_normal");
    vertex_uv_ = glGetAttribLocation(program_, "vertex_uv");
    MVP_ = glGetUniformLocation(program_, "MVP");
    V_ = glGetUniformLocation(program_, "V");
    M_ = glGetUniformLocation(program_, "M");
    light_position_world_ =
            glGetUniformLocation(program_, "light_position_world_4");
    light_color_ = glGetUniformLocation(program_, "light_color_4");
    light_diffuse_power_ =
            glGetUniformLocation(program_, "light_diffuse_power_4");
    light_specular_power_ =
            glGetUniformLocation(program_, "light_specular_power_4");
    light_specular_shininess_ =
            glGetUniformLocation(program_, "light_specular_shininess_4");
    light_ambient_ = glGetUniformLocation(program_, "light_ambient");

    diffuse_texture_ = glGetUniformLocation(program_, "diffuse_texture");
    return true;
}

void TexturePhongShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool TexturePhongShader::BindGeometry(const geometry::Geometry &geometry,
                                      const RenderOption &option,
                                      const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace UnbindGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Prepare data to be passed to GPU
    const size_t num_data_size = GetDataSize(geometry);

    // Create buffers and bind the geometry
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, num_data_size * sizeof(Eigen::Vector3f), 0, GL_STATIC_DRAW);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[0], vertex_position_buffer_, cudaGraphicsMapFlagsNone));
    glGenBuffers(1, &vertex_normal_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    glBufferData(GL_ARRAY_BUFFER, num_data_size * sizeof(Eigen::Vector3f), 0, GL_STATIC_DRAW);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[1], vertex_normal_buffer_, cudaGraphicsMapFlagsNone));
    glGenBuffers(1, &vertex_uv_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffer_);
    glBufferData(GL_ARRAY_BUFFER, num_data_size * sizeof(Eigen::Vector2f), 0, GL_STATIC_DRAW);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[2], vertex_uv_buffer_, cudaGraphicsMapFlagsNone));

    Eigen::Vector3f* raw_points_ptr;
    Eigen::Vector3f* raw_normals_ptr;
    Eigen::Vector2f* raw_uvs_ptr;
    size_t n_bytes;
    cudaSafeCall(cudaGraphicsMapResources(3, cuda_graphics_resources_));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_points_ptr, &n_bytes, cuda_graphics_resources_[0]));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_normals_ptr, &n_bytes, cuda_graphics_resources_[1]));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_uvs_ptr, &n_bytes, cuda_graphics_resources_[2]));
    thrust::device_ptr<Eigen::Vector3f> dev_points_ptr = thrust::device_pointer_cast(raw_points_ptr);
    thrust::device_ptr<Eigen::Vector3f> dev_normals_ptr = thrust::device_pointer_cast(raw_normals_ptr);
    thrust::device_ptr<Eigen::Vector2f> dev_uvs_ptr = thrust::device_pointer_cast(raw_uvs_ptr);

    if (PrepareBinding(geometry, option, view, dev_points_ptr, dev_normals_ptr, dev_uvs_ptr) == false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }
    Unmap(3);
    bound_ = true;
    return true;
}

bool TexturePhongShader::RenderGeometry(const geometry::Geometry &geometry,
                                        const RenderOption &option,
                                        const ViewControl &view) {
    if (PrepareRendering(geometry, option, view) == false) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
    glUniformMatrix4fv(V_, 1, GL_FALSE, view.GetViewMatrix().data());
    glUniformMatrix4fv(M_, 1, GL_FALSE, view.GetModelMatrix().data());
    glUniformMatrix4fv(light_position_world_, 1, GL_FALSE,
                       light_position_world_data_.data());
    glUniformMatrix4fv(light_color_, 1, GL_FALSE, light_color_data_.data());
    glUniform4fv(light_diffuse_power_, 1, light_diffuse_power_data_.data());
    glUniform4fv(light_specular_power_, 1, light_specular_power_data_.data());
    glUniform4fv(light_specular_shininess_, 1,
                 light_specular_shininess_data_.data());
    glUniform4fv(light_ambient_, 1, light_ambient_data_.data());

    glUniform1i(diffuse_texture_, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, diffuse_texture_buffer_);

    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(vertex_normal_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    glVertexAttribPointer(vertex_normal_, 3, GL_FLOAT, GL_FALSE, 0, NULL);

    glEnableVertexAttribArray(vertex_uv_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_uv_buffer_);
    glVertexAttribPointer(vertex_uv_, 2, GL_FLOAT, GL_FALSE, 0, NULL);

    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);

    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_normal_);
    glDisableVertexAttribArray(vertex_uv_);
    return true;
}

void TexturePhongShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_normal_buffer_);
        glDeleteBuffers(1, &vertex_uv_buffer_);
        glDeleteTextures(1, &diffuse_texture_buffer_);
        bound_ = false;
    }
}

void TexturePhongShader::SetLighting(const ViewControl &view,
                                     const RenderOption &option) {
    const auto &box = view.GetBoundingBox();
    light_position_world_data_.setOnes();
    light_color_data_.setOnes();
    for (int i = 0; i < 4; i++) {
        light_position_world_data_.block<3, 1>(0, i) =
                box.GetCenter().cast<GLfloat>() +
                (float)box.GetMaxExtent() *
                        ((float)option.light_position_relative_[i](0) *
                                 view.GetRight() +
                         (float)option.light_position_relative_[i](1) *
                                 view.GetUp() +
                         (float)option.light_position_relative_[i](2) *
                                 view.GetFront());
        light_color_data_.block<3, 1>(0, i) =
                option.light_color_[i].cast<GLfloat>();
    }
    if (option.light_on_) {
        light_diffuse_power_data_ =
                Eigen::Vector4f(option.light_diffuse_power_).cast<GLfloat>();
        light_specular_power_data_ =
                Eigen::Vector4f(option.light_specular_power_).cast<GLfloat>();
        light_specular_shininess_data_ =
                Eigen::Vector4f(option.light_specular_shininess_)
                        .cast<GLfloat>();
        light_ambient_data_.block<3, 1>(0, 0) =
                option.light_ambient_color_.cast<GLfloat>();
        light_ambient_data_(3) = 1.0f;
    } else {
        light_diffuse_power_data_ = gl_helper::GLVector4f::Zero();
        light_specular_power_data_ = gl_helper::GLVector4f::Zero();
        light_specular_shininess_data_ = gl_helper::GLVector4f::Ones();
        light_ambient_data_ = gl_helper::GLVector4f(1.0f, 1.0f, 1.0f, 1.0f);
    }
}

bool TexturePhongShaderForTriangleMesh::PrepareRendering(
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
    SetLighting(view, option);
    return true;
}

bool TexturePhongShaderForTriangleMesh::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector3f> &normals,
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
    if (mesh.HasTriangleNormals() == false ||
        mesh.HasVertexNormals() == false) {
        PrintShaderWarning("Binding failed because mesh has no normals.");
        PrintShaderWarning("Call ComputeVertexNormals() before binding.");
        return false;
    }
    copy_trianglemesh_functor func(thrust::raw_pointer_cast(mesh.vertices_.data()),
                                   thrust::raw_pointer_cast(mesh.vertex_normals_.data()),
                                   (int*)(thrust::raw_pointer_cast(mesh.triangles_.data())),
                                   thrust::raw_pointer_cast(mesh.triangle_normals_.data()),
                                   thrust::raw_pointer_cast(mesh.triangle_uvs_.data()),
                                   option.mesh_shade_option_);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(mesh.triangles_.size() * 3),
                      make_tuple_iterator(points, normals, uvs), func);

    glGenTextures(1, &diffuse_texture_);
    glBindTexture(GL_TEXTURE_2D, diffuse_texture_buffer_);

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

size_t TexturePhongShaderForTriangleMesh::GetDataSize(const geometry::Geometry &geometry) const {
    return ((const geometry::TriangleMesh &)geometry).triangles_.size() * 3;
}