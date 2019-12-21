#include "cupoch/visualization/shader/simple_shader.h"

#include "cupoch/geometry/bounding_volume.h"
#include "cupoch/geometry/lineset.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/visualization/shader/shader.h"
#include "cupoch/visualization/visualizer/render_option.h"
#include "cupoch/visualization/utility/color_map.h"
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
    const thrust::device_ptr<const ColorMap> global_color_map_ = GetGlobalColorMap();
    __device__
    thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> operator() (const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f>& pt_cl) {
        const Eigen::Vector3f &point = thrust::get<0>(pt_cl);
        const Eigen::Vector3f &color = thrust::get<1>(pt_cl);
        Eigen::Vector3f color_tmp;
        switch (color_option_) {
            case RenderOption::PointColorOption::XCoordinate:
                color_tmp = global_color_map_.get()->GetColor(
                            view_.GetBoundingBox().GetXPercentage(point(0)));
                break;
            case RenderOption::PointColorOption::YCoordinate:
                color_tmp = global_color_map_.get()->GetColor(
                            view_.GetBoundingBox().GetYPercentage(point(1)));
                break;
            case RenderOption::PointColorOption::ZCoordinate:
                color_tmp = global_color_map_.get()->GetColor(
                            view_.GetBoundingBox().GetZPercentage(point(2)));
                break;
            case RenderOption::PointColorOption::Color:
            case RenderOption::PointColorOption::Default:
            default:
                if (has_colors_) {
                    color_tmp = color;
                } else {
                    color_tmp = global_color_map_.get()->GetColor(
                                view_.GetBoundingBox().GetZPercentage(point(2)));
                }
                break;
        }
        return thrust::make_tuple(point, color_tmp);
    }
};

struct copy_lineset_functor {
    copy_lineset_functor(const thrust::pair<Eigen::Vector3f, Eigen::Vector3f>* line_coords,
                         const Eigen::Vector3f* line_colors,
                         Eigen::Vector3f* points, Eigen::Vector3f* colors, bool has_colors)
        : line_coords_(line_coords), points_(points), colors_(colors), has_colors_(has_colors) {};
    const thrust::pair<Eigen::Vector3f, Eigen::Vector3f>* line_coords_;
    const Eigen::Vector3f* line_colors_;
    Eigen::Vector3f* points_;
    Eigen::Vector3f* colors_;
    const bool has_colors_;
    __device__
    void operator() (size_t idx) {
        points_[idx * 2] = line_coords_[idx].first;
        points_[idx * 2 + 1] = line_coords_[idx].second;
        Eigen::Vector3f color_tmp;
        if (has_colors_) {
            color_tmp = line_colors_[idx];
        } else {
            color_tmp = Eigen::Vector3f::Zero();
        }
        colors_[idx * 2] = colors_[idx * 2 + 1] = color_tmp;
    }
};

struct line_coordinates_functor {
    line_coordinates_functor(const Eigen::Vector3f* points) : points_(points) {};
    const Eigen::Vector3f* points_;
    __device__
    thrust::pair<Eigen::Vector3f, Eigen::Vector3f> operator() (const Eigen::Vector2i& idxs) const {
        return thrust::make_pair(points_[idxs[0]], points_[idxs[1]]);
    }
};

struct copy_trianglemesh_functor {
    copy_trianglemesh_functor(const Eigen::Vector3f* vertices, const Eigen::Vector3i* triangles,
                              const Eigen::Vector3f* vertex_colors,
                              Eigen::Vector3f* points, Eigen::Vector3f* colors,
                              bool has_vertex_colors, RenderOption::MeshColorOption color_option,
                              const Eigen::Vector3f& default_mesh_color, const ViewControl& view)
                              : vertices_(vertices), triangles_(triangles), vertex_colors_(vertex_colors),
                                points_(points), colors_(colors), has_vertex_colors_(has_vertex_colors),
                                color_option_(color_option), default_mesh_color_(default_mesh_color), view_(view) {};
    const Eigen::Vector3f* vertices_;
    const Eigen::Vector3i* triangles_;
    const Eigen::Vector3f* vertex_colors_;
    Eigen::Vector3f* points_;
    Eigen::Vector3f* colors_;
    const bool has_vertex_colors_;
    const RenderOption::MeshColorOption color_option_;
    const Eigen::Vector3f default_mesh_color_;
    const ViewControl view_;
    const thrust::device_ptr<const ColorMap> global_color_map_ = GetGlobalColorMap();
    __device__
    void operator() (size_t idx) {
        const auto &triangle = triangles_[idx];
        for (size_t j = 0; j < 3; j++) {
            size_t k = idx * 3 + j;
            size_t vi = triangle(j);
            const auto& vertex = vertices_[vi];
            points_[k] = vertex;

            Eigen::Vector3f color_tmp;
            switch (color_option_) {
                case RenderOption::MeshColorOption::XCoordinate:
                    color_tmp = global_color_map_.get()->GetColor(
                                view_.GetBoundingBox().GetXPercentage(vertex(0)));
                    break;
                case RenderOption::MeshColorOption::YCoordinate:
                    color_tmp = global_color_map_.get()->GetColor(
                                view_.GetBoundingBox().GetYPercentage(vertex(1)));
                    break;
                case RenderOption::MeshColorOption::ZCoordinate:
                    color_tmp = global_color_map_.get()->GetColor(
                                view_.GetBoundingBox().GetZPercentage(vertex(2)));
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
            colors_[k] = color_tmp;
        }
    }

};

}

bool SimpleShader::Compile() {
    if (CompileShaders(simple_vertex_shader, NULL, simple_fragment_shader) ==
        false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_color_ = glGetAttribLocation(program_, "vertex_color");
    MVP_ = glGetUniformLocation(program_, "MVP");
    return true;
}

void SimpleShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool SimpleShader::BindGeometry(const geometry::Geometry &geometry,
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
    thrust::device_vector<Eigen::Vector3f> points;
    thrust::device_vector<Eigen::Vector3f> colors;
    if (PrepareBinding(geometry, option, view, points, colors) == false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    // Create buffers and bind the geometry
    cudaGraphicsGLRegisterBuffer(&cuda_graphics_resource_position_, vertex_position_buffer_, cudaGraphicsMapFlagsNone);
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(Eigen::Vector3f),
                 thrust::raw_pointer_cast(points.data()), GL_STATIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&cuda_graphics_resource_color_, vertex_color_buffer_, cudaGraphicsMapFlagsNone);
    glGenBuffers(1, &vertex_color_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Eigen::Vector3f),
                 thrust::raw_pointer_cast(colors.data()), GL_STATIC_DRAW);
    bound_ = true;
    return true;
}

bool SimpleShader::RenderGeometry(const geometry::Geometry &geometry,
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
    glEnableVertexAttribArray(vertex_color_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glVertexAttribPointer(vertex_color_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_color_);
    return true;
}

void SimpleShader::UnbindGeometry() {
    if (bound_) {
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_color_buffer_);
        bound_ = false;
    }
}

bool SimpleShaderForPointCloud::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::PointCloud) {
        PrintShaderWarning("Rendering type is not geometry::PointCloud.");
        return false;
    }
    glPointSize(GLfloat(option.point_size_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForPointCloud::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_vector<Eigen::Vector3f> &points,
        thrust::device_vector<Eigen::Vector3f> &colors) {
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
    points.resize(pointcloud.points_.size());
    colors.resize(pointcloud.points_.size());
    copy_pointcloud_functor func(pointcloud.HasColors(), option.point_color_option_, view);
    thrust::transform(make_tuple_iterator(pointcloud.points_.begin(), pointcloud.colors_.begin()),
                      make_tuple_iterator(pointcloud.points_.end(), pointcloud.colors_.end()),
                      make_tuple_iterator(points.begin(), colors.begin()), func);
    draw_arrays_mode_ = GL_POINTS;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForAxisAlignedBoundingBox::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::AxisAlignedBoundingBox) {
        PrintShaderWarning(
                "Rendering type is not geometry::AxisAlignedBoundingBox.");
        return false;
    }
    glLineWidth(GLfloat(option.line_width_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForAxisAlignedBoundingBox::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_vector<Eigen::Vector3f> &points,
        thrust::device_vector<Eigen::Vector3f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::AxisAlignedBoundingBox) {
        PrintShaderWarning(
                "Rendering type is not geometry::AxisAlignedBoundingBox.");
        return false;
    }
    auto lineset = geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            (const geometry::AxisAlignedBoundingBox &)geometry);
    points.resize(lineset->lines_.size() * 2);
    colors.resize(lineset->lines_.size() * 2);
    thrust::device_vector<thrust::pair<Eigen::Vector3f, Eigen::Vector3f>> line_coords(lineset->lines_.size());
    line_coordinates_functor func_line(thrust::raw_pointer_cast(lineset->points_.data()));
    thrust::transform(lineset->lines_.begin(), lineset->lines_.end(),
                      line_coords.begin(), func_line);
    copy_lineset_functor func_cp(thrust::raw_pointer_cast(line_coords.data()),
                                 thrust::raw_pointer_cast(lineset->colors_.data()),
                                 thrust::raw_pointer_cast(points.data()),
                                 thrust::raw_pointer_cast(colors.data()), lineset->HasColors());
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(lineset->lines_.size()), func_cp);
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}

bool SimpleShaderForTriangleMesh::PrepareRendering(
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

bool SimpleShaderForTriangleMesh::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_vector<Eigen::Vector3f> &points,
        thrust::device_vector<Eigen::Vector3f> &colors) {
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
    points.resize(mesh.triangles_.size() * 3);
    colors.resize(mesh.triangles_.size() * 3);

    copy_trianglemesh_functor func(thrust::raw_pointer_cast(mesh.vertices_.data()),
                                   thrust::raw_pointer_cast(mesh.triangles_.data()),
                                   thrust::raw_pointer_cast(mesh.vertex_colors_.data()),
                                   thrust::raw_pointer_cast(points.data()),
                                   thrust::raw_pointer_cast(colors.data()),
                                   mesh.HasVertexColors(), option.mesh_color_option_,
                                   option.default_mesh_color_, view);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(mesh.triangles_.size()), func);
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(points.size());
    return true;
}
