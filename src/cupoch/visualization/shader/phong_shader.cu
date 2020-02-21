#include "cupoch/visualization/shader/phong_shader.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/visualization/shader/shader.h"
#include "cupoch/visualization/utility/color_map.h"
#include <thrust/iterator/constant_iterator.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace cupoch;
using namespace cupoch::visualization;
using namespace cupoch::visualization::glsl;

namespace {
struct copy_pointcloud_functor{
    copy_pointcloud_functor(bool has_colors, RenderOption::PointColorOption color_option, const ViewControl& view)
        : has_colors_(has_colors), color_option_(color_option), view_(view) {};
    const bool has_colors_;
    const RenderOption::PointColorOption color_option_;
    const ViewControl view_;
    const ColorMap::ColorMapOption colormap_option_ = GetGlobalColorMapOption();
    __device__
    thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f> operator() (const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>& pt_nm_cl) {
        const Eigen::Vector3f &point = thrust::get<0>(pt_nm_cl);
        const Eigen::Vector3f &normal = thrust::get<1>(pt_nm_cl);
        const Eigen::Vector3f &color = thrust::get<2>(pt_nm_cl);
        Eigen::Vector3f color_tmp;
        switch (color_option_) {
            case RenderOption::PointColorOption::XCoordinate:
                color_tmp = GetColorMapColor(view_.GetBoundingBox().GetXPercentage(point(0)), colormap_option_);
                break;
            case RenderOption::PointColorOption::YCoordinate:
                color_tmp = GetColorMapColor(view_.GetBoundingBox().GetYPercentage(point(1)), colormap_option_);
                break;
            case RenderOption::PointColorOption::ZCoordinate:
                color_tmp = GetColorMapColor(view_.GetBoundingBox().GetZPercentage(point(2)), colormap_option_);
                break;
            case RenderOption::PointColorOption::Color:
            case RenderOption::PointColorOption::Default:
            default:
                if (has_colors_) {
                    color_tmp = color;
                } else {
                    color_tmp = GetColorMapColor(view_.GetBoundingBox().GetZPercentage(point(2)), colormap_option_);
                }
                break;
        }
        return thrust::make_tuple(point, normal, color_tmp);
    }
};

struct copy_trianglemesh_functor {
    copy_trianglemesh_functor(const Eigen::Vector3f* vertices, const int* triangles,
                              const Eigen::Vector3f* triangle_normals, const Eigen::Vector3f* vertex_normals,
                              const Eigen::Vector3f* vertex_colors,
                              bool has_vertex_colors, RenderOption::MeshColorOption color_option,
                              RenderOption::MeshShadeOption shade_option, const Eigen::Vector3f& default_mesh_color,
                              const ViewControl& view)
                              : vertices_(vertices), triangles_(triangles), triangle_normals_(triangle_normals),
                                vertex_normals_(vertex_normals), vertex_colors_(vertex_colors), has_vertex_colors_(has_vertex_colors),
                                color_option_(color_option), shade_option_(shade_option),
                                default_mesh_color_(default_mesh_color), view_(view) {};
    const Eigen::Vector3f* vertices_;
    const int* triangles_;
    const Eigen::Vector3f* triangle_normals_;
    const Eigen::Vector3f* vertex_normals_;
    const Eigen::Vector3f* vertex_colors_;
    const bool has_vertex_colors_;
    const RenderOption::MeshColorOption color_option_;
    const RenderOption::MeshShadeOption shade_option_;
    const Eigen::Vector3f default_mesh_color_;
    const ViewControl view_;
    const ColorMap::ColorMapOption colormap_option_ = GetGlobalColorMapOption();
    __device__
    thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f> operator() (size_t k) const {
        int idx = k / 3;
        int vi = triangles_[k];
        const Eigen::Vector3f &vertex = vertices_[vi];

        Eigen::Vector3f color_tmp;
        switch (color_option_) {
            case RenderOption::MeshColorOption::XCoordinate:
                color_tmp = GetColorMapColor(view_.GetBoundingBox().GetXPercentage(vertex(0)), colormap_option_);
                break;
            case RenderOption::MeshColorOption::YCoordinate:
                color_tmp = GetColorMapColor(view_.GetBoundingBox().GetYPercentage(vertex(1)), colormap_option_);
                break;
            case RenderOption::MeshColorOption::ZCoordinate:
                color_tmp = GetColorMapColor(view_.GetBoundingBox().GetZPercentage(vertex(2)), colormap_option_);
                break;
            case RenderOption::MeshColorOption::Color:
                if (has_vertex_colors_) {
                    color_tmp = vertex_colors_[vi];
                    break;
                }
            case RenderOption::MeshColorOption::Default:
            default:
                color_tmp = default_mesh_color_;
                break;
        }

        if (shade_option_ ==
            RenderOption::MeshShadeOption::FlatShade) {
            return thrust::make_tuple(vertex, triangle_normals_[idx], color_tmp);
        } else {
            return thrust::make_tuple(vertex, vertex_normals_[vi], color_tmp);
        }
    }

};

}

bool PhongShader::Compile() {
    if (CompileShaders(phong_vertex_shader, NULL, phong_fragment_shader) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_normal_ = glGetAttribLocation(program_, "vertex_normal");
    vertex_color_ = glGetAttribLocation(program_, "vertex_color");
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
    return true;
}

void PhongShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool PhongShader::BindGeometry(const geometry::Geometry &geometry,
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
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[0], vertex_position_buffer_, cudaGraphicsMapFlagsNone));
    glGenBuffers(1, &vertex_normal_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    glBufferData(GL_ARRAY_BUFFER, num_data_size * sizeof(Eigen::Vector3f), 0, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[1], vertex_normal_buffer_, cudaGraphicsMapFlagsNone));
    glGenBuffers(1, &vertex_color_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glBufferData(GL_ARRAY_BUFFER, num_data_size * sizeof(Eigen::Vector3f), 0, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[2], vertex_color_buffer_, cudaGraphicsMapFlagsNone));

    Eigen::Vector3f* raw_points_ptr;
    Eigen::Vector3f* raw_normals_ptr;
    Eigen::Vector3f* raw_colors_ptr;
    size_t n_bytes;
    cudaSafeCall(cudaGraphicsMapResources(3, cuda_graphics_resources_));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_points_ptr, &n_bytes, cuda_graphics_resources_[0]));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_normals_ptr, &n_bytes, cuda_graphics_resources_[1]));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_colors_ptr, &n_bytes, cuda_graphics_resources_[2]));
    thrust::device_ptr<Eigen::Vector3f> dev_points_ptr = thrust::device_pointer_cast(raw_points_ptr);
    thrust::device_ptr<Eigen::Vector3f> dev_normals_ptr = thrust::device_pointer_cast(raw_normals_ptr);
    thrust::device_ptr<Eigen::Vector3f> dev_colors_ptr = thrust::device_pointer_cast(raw_colors_ptr);

    if (PrepareBinding(geometry, option, view, dev_points_ptr, dev_normals_ptr, dev_colors_ptr) ==
        false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    Unmap(3);
    bound_ = true;
    return true;
}

bool PhongShader::RenderGeometry(const geometry::Geometry &geometry,
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
    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(vertex_normal_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    glVertexAttribPointer(vertex_normal_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(vertex_color_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_normal_);
    glDisableVertexAttribArray(vertex_color_);
    return true;
}

void PhongShader::UnbindGeometry() {
    if (bound_) {
        cudaSafeCall(cudaGraphicsUnregisterResource(cuda_graphics_resources_[0]));
        cudaSafeCall(cudaGraphicsUnregisterResource(cuda_graphics_resources_[1]));
        cudaSafeCall(cudaGraphicsUnregisterResource(cuda_graphics_resources_[2]));
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_normal_buffer_);
        glDeleteBuffers(1, &vertex_color_buffer_);
        bound_ = false;
    }
}

void PhongShader::SetLighting(const ViewControl &view,
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

bool PhongShaderForPointCloud::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloud) {
        PrintShaderWarning("Rendering type is not geometry::PointCloud.");
        return false;
    }
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    glPointSize(GLfloat(option.point_size_));
    SetLighting(view, option);
    return true;
}

bool PhongShaderForPointCloud::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector3f> &normals,
        thrust::device_ptr<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloud) {
        PrintShaderWarning("Rendering type is not geometry::PointCloud.");
        return false;
    }
    const geometry::PointCloud &pointcloud =
            (const geometry::PointCloud &)geometry;
    if (pointcloud.HasPoints() == false) {
        PrintShaderWarning("Binding failed with empty pointcloud.");
        return false;
    }
    if (pointcloud.HasNormals() == false) {
        PrintShaderWarning("Binding failed with pointcloud with no normals.");
        return false;
    }
    copy_pointcloud_functor func(pointcloud.HasColors(), option.point_color_option_, view);
    if (pointcloud.HasColors()) {
        thrust::transform(make_tuple_iterator(pointcloud.points_.begin(), pointcloud.normals_.begin(), pointcloud.colors_.begin()),
                          make_tuple_iterator(pointcloud.points_.end(), pointcloud.normals_.end(), pointcloud.colors_.end()),
                          make_tuple_iterator(points, normals, colors), func);
    } else {
        thrust::transform(make_tuple_iterator(pointcloud.points_.begin(), pointcloud.normals_.begin(),
                                              thrust::constant_iterator<Eigen::Vector3f>(Eigen::Vector3f::Zero())),
                          make_tuple_iterator(pointcloud.points_.end(), pointcloud.normals_.end(),
                                              thrust::constant_iterator<Eigen::Vector3f>(Eigen::Vector3f::Zero())),
                          make_tuple_iterator(points, normals, colors), func);
    }
    draw_arrays_mode_ = GL_POINTS;
    draw_arrays_size_ = GLsizei(pointcloud.points_.size());
    return true;
}

size_t PhongShaderForPointCloud::GetDataSize(const geometry::Geometry &geometry) const {
    return ((const geometry::PointCloud &)geometry).points_.size();
}

bool PhongShaderForTriangleMesh::PrepareRendering(
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

bool PhongShaderForTriangleMesh::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector3f> &normals,
        thrust::device_ptr<Eigen::Vector3f> &colors) {
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
                                   (int*)(thrust::raw_pointer_cast(mesh.triangles_.data())),
                                   thrust::raw_pointer_cast(mesh.triangle_normals_.data()),
                                   thrust::raw_pointer_cast(mesh.vertex_normals_.data()),
                                   thrust::raw_pointer_cast(mesh.vertex_colors_.data()),
                                   mesh.HasVertexColors(), option.mesh_color_option_,
                                   option.mesh_shade_option_,
                                   option.default_mesh_color_, view);
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(mesh.triangles_.size() * 3),
                      make_tuple_iterator(points, normals, colors), func);
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(mesh.triangles_.size() * 3);
    return true;
}

size_t PhongShaderForTriangleMesh::GetDataSize(const geometry::Geometry &geometry) const {
    return ((const geometry::TriangleMesh &)geometry).triangles_.size() * 3;
}