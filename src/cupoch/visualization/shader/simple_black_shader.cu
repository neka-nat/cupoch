#include "cupoch/visualization/shader/simple_black_shader.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/visualization/shader/shader.h"
#include "cupoch/visualization/utility/color_map.h"

using namespace cupoch;
using namespace cupoch::visualization;
using namespace cupoch::visualization::glsl;

namespace {

struct copy_pointcloud_normal_functor {
    copy_pointcloud_normal_functor(const Eigen::Vector3f* points,
                                   const Eigen::Vector3f* normals, float line_length)
                                   : points_(points), normals_(normals), line_length_(line_length) {};
    const Eigen::Vector3f* points_;
    const Eigen::Vector3f* normals_;
    const float line_length_;
    __device__
    Eigen::Vector3f operator() (size_t idx) {
        int i = idx / 2;
        int j = idx % 2;
        if (j == 0) {
            return points_[i];
        } else {
            return points_[i] + normals_[i] * line_length_;
        }
    }
};

struct copy_mesh_wireflame_functor {
    copy_mesh_wireflame_functor(const Eigen::Vector3f* vertices, const int* triangles)
                                : vertices_(vertices), triangles_(triangles) {};
    const Eigen::Vector3f* vertices_;
    const int* triangles_;
    __device__
    Eigen::Vector3f operator() (size_t k) {
        int vi = triangles_[k];
        return vertices_[vi];
    }
};

}

bool SimpleBlackShader::Compile() {
    if (CompileShaders(simple_black_vertex_shader, NULL,
                       simple_black_fragment_shader) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void SimpleBlackShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool SimpleBlackShader::BindGeometry(const geometry::Geometry &geometry,
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
    thrust::device_vector<Eigen::Vector3f> points;
    if (PrepareBinding(geometry, option, view, points) == false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Eigen::Vector3f),
                 thrust::raw_pointer_cast(points.data()), GL_STATIC_DRAW);

    bound_ = true;
    return true;
}

bool SimpleBlackShader::RenderGeometry(const geometry::Geometry &geometry,
                                       const RenderOption &option,
                                       const ViewControl &view) {
    if (PrepareRendering(geometry, option, view) == false) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }
    glUseProgram(program_);
    glUniformMatrix4fv(MVP_, 1, GL_FALSE, view.GetMVPMatrix().data());
    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    return true;
}

void SimpleBlackShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        bound_ = false;
    }
}

bool SimpleBlackShaderForPointCloudNormal::PrepareRendering(
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
    return true;
}

bool SimpleBlackShaderForPointCloudNormal::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_vector<Eigen::Vector3f> &points) {
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
    points.resize(pointcloud.points_.size() * 2);
    float line_length =
            option.point_size_ * 0.01 * view.GetBoundingBox().GetMaxExtent();
    copy_pointcloud_normal_functor func(thrust::raw_pointer_cast(pointcloud.points_.data()),
                                        thrust::raw_pointer_cast(pointcloud.normals_.data()), line_length);
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(pointcloud.points_.size() * 2),
                      points.begin(), func);
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleBlackShaderForTriangleMeshWireFrame::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::TriangleMesh) {
        PrintShaderWarning("Rendering type is not geometry::TriangleMesh.");
        return false;
    }
    glLineWidth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDisable(GL_POLYGON_OFFSET_FILL);
    return true;
}

bool SimpleBlackShaderForTriangleMeshWireFrame::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_vector<Eigen::Vector3f> &points) {
    if (geometry.GetGeometryType() !=
                geometry::Geometry::GeometryType::TriangleMesh) {
        PrintShaderWarning("Rendering type is not geometry::TriangleMesh.");
        return false;
    }
    const geometry::TriangleMesh &mesh =
            (const geometry::TriangleMesh &)geometry;
    if (mesh.HasTriangles() == false) {
        PrintShaderWarning("Binding failed with empty geometry::TriangleMesh.");
        return false;
    }
    points.resize(mesh.triangles_.size() * 3);
    copy_mesh_wireflame_functor func(thrust::raw_pointer_cast(mesh.vertices_.data()),
                                     (int*)(thrust::raw_pointer_cast(mesh.triangles_.data())));
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(mesh.triangles_.size() * 3),
                      points.begin(), func);
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}