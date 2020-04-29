#include "cupoch/visualization/shader/normal_shader.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/visualization/shader/shader.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace cupoch;
using namespace cupoch::visualization;
using namespace cupoch::visualization::glsl;

namespace {

struct copy_trianglemesh_functor {
    copy_trianglemesh_functor(const Eigen::Vector3f* vertices, const Eigen::Vector3f* vertex_normals,
                              const int* triangles, const Eigen::Vector3f* triangle_normals,
                              RenderOption::MeshShadeOption shade_option)
                              : vertices_(vertices), vertex_normals_(vertex_normals),
                                triangles_(triangles), triangle_normals_(triangle_normals),
                                shade_option_(shade_option) {};
    const Eigen::Vector3f* vertices_;
    const Eigen::Vector3f* vertex_normals_;
    const int* triangles_;
    const Eigen::Vector3f* triangle_normals_;
    const RenderOption::MeshShadeOption shade_option_;
    __device__
    thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> operator() (size_t k) const {
        int i = k / 3;
        int vi = triangles_[k];
        const auto &vertex = vertices_[vi];
        return (shade_option_ == RenderOption::MeshShadeOption::FlatShade) ? thrust::make_tuple(vertex, triangle_normals_[i]) :
            thrust::make_tuple(vertex, vertex_normals_[vi]);
    }
};

}

bool NormalShader::Compile() {
    if (CompileShaders(normal_vertex_shader, NULL, normal_fragment_shader) ==
        false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_normal_ = glGetAttribLocation(program_, "vertex_normal");
    MVP_ = glGetUniformLocation(program_, "MVP");
    V_ = glGetUniformLocation(program_, "V");
    M_ = glGetUniformLocation(program_, "M");
    return true;
}

void NormalShader::Release() {
    UnbindGeometry(true);
    ReleaseProgram();
}

bool NormalShader::BindGeometry(const geometry::Geometry &geometry,
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

    Eigen::Vector3f* raw_points_ptr;
    Eigen::Vector3f* raw_normals_ptr;
    size_t n_bytes;
    cudaSafeCall(cudaGraphicsMapResources(2, cuda_graphics_resources_));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_points_ptr, &n_bytes, cuda_graphics_resources_[0]));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_normals_ptr, &n_bytes, cuda_graphics_resources_[1]));
    thrust::device_ptr<Eigen::Vector3f> dev_points_ptr = thrust::device_pointer_cast(raw_points_ptr);
    thrust::device_ptr<Eigen::Vector3f> dev_normals_ptr = thrust::device_pointer_cast(raw_normals_ptr);

    if (PrepareBinding(geometry, option, view, dev_points_ptr, dev_normals_ptr) == false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    Unmap(2);
    bound_ = true;
    return true;
}

bool NormalShader::RenderGeometry(const geometry::Geometry &geometry,
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
    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(vertex_normal_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_normal_buffer_);
    glVertexAttribPointer(vertex_normal_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_normal_);
    return true;
}

void NormalShader::UnbindGeometry(bool finalize) {
    if (bound_) {
        if (!finalize) {
            cudaSafeCall(cudaGraphicsUnregisterResource(cuda_graphics_resources_[0]));
            cudaSafeCall(cudaGraphicsUnregisterResource(cuda_graphics_resources_[1]));
        }
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_normal_buffer_);
        bound_ = false;
    }
}

bool NormalShaderForPointCloud::PrepareRendering(
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
    return true;
}

bool NormalShaderForPointCloud::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector3f> &normals) {
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
    thrust::copy(pointcloud.points_.begin(), pointcloud.points_.end(), points);
    thrust::copy(pointcloud.normals_.begin(), pointcloud.normals_.end(), normals);
    draw_arrays_mode_ = GL_POINTS;
    draw_arrays_size_ = GLsizei(pointcloud.points_.size());
    return true;
}

size_t NormalShaderForPointCloud::GetDataSize(const geometry::Geometry &geometry) const {
    return ((const geometry::PointCloud &)geometry).points_.size();
}

bool NormalShaderForTriangleMesh::PrepareRendering(
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

bool NormalShaderForTriangleMesh::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector3f> &normals) {
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
                                   option.mesh_shade_option_);
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(mesh.triangles_.size() * 3),
                      make_tuple_iterator(points, normals), func);
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(mesh.triangles_.size() * 3);
    return true;
}

size_t NormalShaderForTriangleMesh::GetDataSize(const geometry::Geometry &geometry) const {
    return ((const geometry::TriangleMesh &)geometry).triangles_.size() * 3;
}