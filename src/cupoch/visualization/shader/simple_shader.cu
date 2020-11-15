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
#include <cuda_runtime.h>
#include <thrust/iterator/constant_iterator.h>

#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/graph.h"
#include "cupoch/geometry/lineset.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/geometry/distancetransform.h"
#include "cupoch/geometry/geometry_functor.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/utility/platform.h"
#include "cupoch/utility/range.h"
#include "cupoch/visualization/shader/shader.h"
#include "cupoch/visualization/shader/simple_shader.h"
#include "cupoch/visualization/utility/color_map.h"
#include "cupoch/visualization/visualizer/render_option.h"

using namespace cupoch;
using namespace cupoch::visualization;
using namespace cupoch::visualization::glsl;

namespace {

// Vertex indices of 12 lines in a cuboid
__constant__ int cuboid_lines_vertex_indices[12][2] = {
        {0, 1}, {0, 2}, {0, 4}, {3, 1}, {3, 2}, {3, 7},
        {5, 1}, {5, 4}, {5, 7}, {6, 2}, {6, 4}, {6, 7},
};

template <int Dim>
struct copy_pointcloud_functor {
    copy_pointcloud_functor(bool has_colors,
                            RenderOption::PointColorOption color_option,
                            const ViewControl &view)
        : has_colors_(has_colors), color_option_(color_option), view_(view){};
    const bool has_colors_;
    const RenderOption::PointColorOption color_option_;
    const ViewControl view_;
    const ColorMap::ColorMapOption colormap_option_ = GetGlobalColorMapOption();
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector4f> operator()(
            const thrust::tuple<Eigen::Matrix<float, Dim, 1>, Eigen::Vector3f> &pt_cl);

    __device__ Eigen::Vector4f GetColor(const Eigen::Vector3f& point,
                                        const Eigen::Vector3f& color) const {
        Eigen::Vector4f color_tmp;
        color_tmp[3] = 1.0;
        switch (color_option_) {
            case RenderOption::PointColorOption::XCoordinate:
                color_tmp.head<3>() = GetColorMapColor(
                        view_.GetBoundingBox().GetXPercentage(point(0)),
                        colormap_option_);
                break;
            case RenderOption::PointColorOption::YCoordinate:
                color_tmp.head<3>() = GetColorMapColor(
                        view_.GetBoundingBox().GetYPercentage(point(1)),
                        colormap_option_);
                break;
            case RenderOption::PointColorOption::ZCoordinate:
                color_tmp.head<3>() = GetColorMapColor(
                        view_.GetBoundingBox().GetZPercentage(point(2)),
                        colormap_option_);
                break;
            case RenderOption::PointColorOption::Color:
            case RenderOption::PointColorOption::Default:
            default:
                if (has_colors_) {
                    color_tmp.head<3>() = color;
                } else {
                    color_tmp.head<3>() = GetColorMapColor(
                            view_.GetBoundingBox().GetZPercentage(point(2)),
                            colormap_option_);
                }
                break;
        }
        return color_tmp;
    }
};

template <>
__device__
thrust::tuple<Eigen::Vector3f, Eigen::Vector4f> copy_pointcloud_functor<3>::operator()(
            const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> &pt_cl) {
    const Eigen::Vector3f &point = thrust::get<0>(pt_cl);
    const Eigen::Vector3f &color = thrust::get<1>(pt_cl);
    return thrust::make_tuple(point, GetColor(point, color));
}

template <>
__device__
thrust::tuple<Eigen::Vector3f, Eigen::Vector4f> copy_pointcloud_functor<2>::operator()(
            const thrust::tuple<Eigen::Vector2f, Eigen::Vector3f> &pt_cl) {
    const Eigen::Vector3f point = (Eigen::Vector3f() << thrust::get<0>(pt_cl), 0.0).finished();
    const Eigen::Vector3f &color = thrust::get<1>(pt_cl);
    return thrust::make_tuple(point, GetColor(point, color));
}


struct copy_lineset_functor {
    copy_lineset_functor(
            const thrust::pair<Eigen::Vector3f, Eigen::Vector3f> *line_coords,
            const Eigen::Vector3f *line_colors,
            bool has_colors)
        : line_coords_(line_coords),
          line_colors_(line_colors),
          has_colors_(has_colors){};
    const thrust::pair<Eigen::Vector3f, Eigen::Vector3f> *line_coords_;
    const Eigen::Vector3f *line_colors_;
    const bool has_colors_;
    const Eigen::Vector3f default_line_color_ = geometry::DEFAULT_LINE_COLOR;
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector4f> operator()(
            size_t k) const {
        int i = k / 2;
        int j = k % 2;
        Eigen::Vector4f color_tmp;
        color_tmp[3]  = 1.0;
        color_tmp.head<3>() =
                (has_colors_) ? line_colors_[i] : default_line_color_;
        if (j == 0) {
            return thrust::make_tuple(line_coords_[i].first, color_tmp);
        } else {
            return thrust::make_tuple(line_coords_[i].second, color_tmp);
        }
    }
};

template <int Dim>
struct line_coordinates_functor {
    line_coordinates_functor(const Eigen::Matrix<float, Dim, 1> *points) : points_(points){};
    const Eigen::Matrix<float, Dim, 1> *points_;
    __device__ thrust::pair<Eigen::Vector3f, Eigen::Vector3f> operator()(
            const Eigen::Vector2i &idxs) const;
};

template <>
__device__
thrust::pair<Eigen::Vector3f, Eigen::Vector3f> line_coordinates_functor<3>::operator()(
    const Eigen::Vector2i &idxs) const {
    return thrust::make_pair(points_[idxs[0]], points_[idxs[1]]);
}

template <>
__device__
thrust::pair<Eigen::Vector3f, Eigen::Vector3f> line_coordinates_functor<2>::operator()(
    const Eigen::Vector2i &idxs) const {
    const Eigen::Vector3f p1 = (Eigen::Vector3f() << points_[idxs[0]], 0.0).finished();
    const Eigen::Vector3f p2 = (Eigen::Vector3f() << points_[idxs[1]], 0.0).finished();
    return thrust::make_pair(p1, p2);
}

struct copy_trianglemesh_functor {
    copy_trianglemesh_functor(const Eigen::Vector3f *vertices,
                              const int *triangles,
                              const Eigen::Vector3f *vertex_colors,
                              bool has_vertex_colors,
                              RenderOption::MeshColorOption color_option,
                              const Eigen::Vector3f &default_mesh_color,
                              const ViewControl &view)
        : vertices_(vertices),
          triangles_(triangles),
          vertex_colors_(vertex_colors),
          has_vertex_colors_(has_vertex_colors),
          color_option_(color_option),
          default_mesh_color_(default_mesh_color),
          view_(view){};
    const Eigen::Vector3f *vertices_;
    const int *triangles_;
    const Eigen::Vector3f *vertex_colors_;
    const bool has_vertex_colors_;
    const RenderOption::MeshColorOption color_option_;
    const Eigen::Vector3f default_mesh_color_;
    const ViewControl view_;
    const ColorMap::ColorMapOption colormap_option_ = GetGlobalColorMapOption();
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector4f> operator()(
            size_t k) const {
        size_t vi = triangles_[k];
        const auto &vertex = vertices_[vi];
        Eigen::Vector4f color_tmp;
        color_tmp[3] = 1.0;
        switch (color_option_) {
            case RenderOption::MeshColorOption::XCoordinate:
                color_tmp.head<3>() = GetColorMapColor(
                        view_.GetBoundingBox().GetXPercentage(vertex(0)),
                        colormap_option_);
                break;
            case RenderOption::MeshColorOption::YCoordinate:
                color_tmp.head<3>() = GetColorMapColor(
                        view_.GetBoundingBox().GetYPercentage(vertex(1)),
                        colormap_option_);
                break;
            case RenderOption::MeshColorOption::ZCoordinate:
                color_tmp.head<3>() = GetColorMapColor(
                        view_.GetBoundingBox().GetZPercentage(vertex(2)),
                        colormap_option_);
                break;
            case RenderOption::MeshColorOption::Color:
                if (has_vertex_colors_) {
                    color_tmp.head<3>() = vertex_colors_[vi];
                    break;
                }
            case RenderOption::MeshColorOption::Default:
            default:
                color_tmp.head<3>() = default_mesh_color_;
                break;
        }
        return thrust::make_tuple(vertex, color_tmp);
    }
};

struct copy_voxelgrid_line_functor {
    copy_voxelgrid_line_functor(const Eigen::Vector3f *vertices,
                                const geometry::Voxel *voxels,
                                bool has_colors,
                                RenderOption::MeshColorOption color_option,
                                const Eigen::Vector3f &default_mesh_color,
                                const ViewControl &view)
        : vertices_(vertices),
          voxels_(voxels),
          has_colors_(has_colors),
          color_option_(color_option),
          default_mesh_color_(default_mesh_color),
          view_(view){};
    const Eigen::Vector3f *vertices_;
    const geometry::Voxel *voxels_;
    const bool has_colors_;
    const RenderOption::MeshColorOption color_option_;
    const Eigen::Vector3f default_mesh_color_;
    const ViewControl view_;
    const ColorMap::ColorMapOption colormap_option_ = GetGlobalColorMapOption();
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector4f> operator()(
            size_t idx) const {
        int i = idx / (12 * 2);
        int jk = idx % (12 * 2);
        int j = jk / 2;
        int k = jk % 2;
        // Voxel color (applied to all points)
        Eigen::Vector4f voxel_color;
        voxel_color[3] = 1.0;
        switch (color_option_) {
            case RenderOption::MeshColorOption::XCoordinate:
                voxel_color.head<3>() =
                        GetColorMapColor(view_.GetBoundingBox().GetXPercentage(
                                                 vertices_[i * 8](0)),
                                         colormap_option_);
                break;
            case RenderOption::MeshColorOption::YCoordinate:
                voxel_color.head<3>() =
                        GetColorMapColor(view_.GetBoundingBox().GetYPercentage(
                                                 vertices_[i * 8](1)),
                                         colormap_option_);
                break;
            case RenderOption::MeshColorOption::ZCoordinate:
                voxel_color.head<3>() =
                        GetColorMapColor(view_.GetBoundingBox().GetZPercentage(
                                                 vertices_[i * 8](2)),
                                         colormap_option_);
                break;
            case RenderOption::MeshColorOption::Color:
                if (has_colors_) {
                    voxel_color.head<3>() = voxels_[i].color_;
                    break;
                }
            case RenderOption::MeshColorOption::Default:
            default:
                voxel_color.head<3>() = default_mesh_color_;
                break;
        }
        return thrust::make_tuple(
                vertices_[i * 8 + cuboid_lines_vertex_indices[j][k]],
                voxel_color);
    }
};

struct copy_distance_voxel_functor {
    copy_distance_voxel_functor(float voxel_size,
                                int resolution,
                                const Eigen::Vector3f& origin,
                                float distance_max)
        : voxel_size_(voxel_size), resolution_(resolution),
        origin_(origin), distance_max_(distance_max){};
    const float voxel_size_;
    const int resolution_;
    const Eigen::Vector3f origin_;
    const float distance_max_;
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector4f>
    operator()(const thrust::tuple<size_t, geometry::DistanceVoxel>& kv) const {
        int idx = thrust::get<0>(kv);
        geometry::DistanceVoxel v = thrust::get<1>(kv);
        int res2 = resolution_ * resolution_;
        int x = idx / res2;
        int yz = idx % res2;
        int y = yz / resolution_;
        int z = yz % resolution_;
        // Voxel color (applied to all points)
        Eigen::Vector4f voxel_color = Eigen::Vector4f::Ones();
        int h_res = resolution_ / 2;
        Eigen::Vector3f pt = (Eigen::Vector3i(x - h_res, y - h_res, z - h_res).cast<float>() + Eigen::Vector3f::Constant(0.5)) * voxel_size_ - origin_;
        voxel_color[3] = 1.0 - min(v.distance_, distance_max_) / distance_max_;
        return thrust::make_tuple(pt, voxel_color);
    }
};

struct alpha_greater_functor {
    __device__ bool operator() (const thrust::tuple<Eigen::Vector3f, Eigen::Vector4f>& lhs,
                                const thrust::tuple<Eigen::Vector3f, Eigen::Vector4f>& rhs) const {
        return thrust::get<1>(lhs)[3] > thrust::get<1>(rhs)[3];
    }
};

}  // namespace

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
    UnbindGeometry(true);
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
    const size_t num_data_size = GetDataSize(geometry);

    // Create buffers and bind the geometry
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, num_data_size * sizeof(Eigen::Vector3f), 0,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[0],
                                              vertex_position_buffer_,
                                              cudaGraphicsMapFlagsNone));
    glGenBuffers(1, &vertex_color_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_color_buffer_);
    glBufferData(GL_ARRAY_BUFFER, num_data_size * sizeof(Eigen::Vector4f), 0,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[1],
                                              vertex_color_buffer_,
                                              cudaGraphicsMapFlagsNone));

    Eigen::Vector3f *raw_points_ptr;
    Eigen::Vector4f *raw_colors_ptr;
    size_t n_bytes;
    cudaSafeCall(cudaGraphicsMapResources(2, cuda_graphics_resources_));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            (void **)&raw_points_ptr, &n_bytes, cuda_graphics_resources_[0]));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer(
            (void **)&raw_colors_ptr, &n_bytes, cuda_graphics_resources_[1]));
    thrust::device_ptr<Eigen::Vector3f> dev_points_ptr =
            thrust::device_pointer_cast(raw_points_ptr);
    thrust::device_ptr<Eigen::Vector4f> dev_colors_ptr =
            thrust::device_pointer_cast(raw_colors_ptr);

    if (PrepareBinding(geometry, option, view, dev_points_ptr,
                       dev_colors_ptr) == false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    Unmap(2);
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
    glVertexAttribPointer(vertex_color_, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_color_);
    return true;
}

void SimpleShader::UnbindGeometry(bool finalize) {
    if (bound_) {
        if (!finalize) {
            cudaSafeCall(cudaGraphicsUnregisterResource(
                    cuda_graphics_resources_[0]));
            cudaSafeCall(cudaGraphicsUnregisterResource(
                    cuda_graphics_resources_[1]));
        }
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
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector4f> &colors) {
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
    copy_pointcloud_functor<3> func(pointcloud.HasColors(),
                                    option.point_color_option_, view);
    if (pointcloud.HasColors()) {
        thrust::transform(
                make_tuple_begin(pointcloud.points_, pointcloud.colors_),
                make_tuple_end(pointcloud.points_, pointcloud.colors_),
                make_tuple_iterator(points, colors), func);
    } else {
        thrust::transform(
                make_tuple_iterator(pointcloud.points_.begin(),
                                    thrust::constant_iterator<Eigen::Vector3f>(
                                            Eigen::Vector3f::Zero())),
                make_tuple_iterator(pointcloud.points_.end(),
                                    thrust::constant_iterator<Eigen::Vector3f>(
                                            Eigen::Vector3f::Zero())),
                make_tuple_iterator(points, colors), func);
    }
    draw_arrays_mode_ = GL_POINTS;
    draw_arrays_size_ = GLsizei(pointcloud.points_.size());
    return true;
}

size_t SimpleShaderForPointCloud::GetDataSize(
        const geometry::Geometry &geometry) const {
    return ((const geometry::PointCloud &)geometry).points_.size();
}

bool SimpleShaderForLineSet::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::LineSet) {
        PrintShaderWarning("Rendering type is not geometry::LineSet.");
        return false;
    }
    glLineWidth(GLfloat(option.line_width_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForLineSet::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector4f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::LineSet) {
        PrintShaderWarning("Rendering type is not geometry::LineSet.");
        return false;
    }
    const geometry::LineSet<3> &lineset =
            (const geometry::LineSet<3> &)geometry;
    if (lineset.HasLines() == false) {
        PrintShaderWarning("Binding failed with empty geometry::LineSet.");
        return false;
    }
    utility::device_vector<thrust::pair<Eigen::Vector3f, Eigen::Vector3f>>
            line_coords(lineset.lines_.size());
    line_coordinates_functor<3> func_line(
            thrust::raw_pointer_cast(lineset.points_.data()));
    thrust::transform(lineset.lines_.begin(), lineset.lines_.end(),
                      line_coords.begin(), func_line);
    copy_lineset_functor func_cp(
            thrust::raw_pointer_cast(line_coords.data()),
            thrust::raw_pointer_cast(lineset.colors_.data()),
            lineset.HasColors());
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(lineset.lines_.size() * 2),
                      make_tuple_iterator(points, colors), func_cp);
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(lineset.lines_.size() * 2);
    return true;
}

size_t SimpleShaderForLineSet::GetDataSize(
        const geometry::Geometry &geometry) const {
    return ((const geometry::LineSet<3> &)geometry).lines_.size() * 2;
}

template <int Dim>
bool SimpleShaderForGraphNode<Dim>::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() != geometry::Geometry::GeometryType::Graph) {
        PrintShaderWarning("Rendering type is not geometry::Graph.");
        return false;
    }
    glPointSize(GLfloat(option.point_size_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

template <int Dim>
bool SimpleShaderForGraphNode<Dim>::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector4f> &colors) {
    if (geometry.GetGeometryType() != geometry::Geometry::GeometryType::Graph) {
        PrintShaderWarning("Rendering type is not geometry::Graph.");
        return false;
    }
    const geometry::Graph<Dim> &graph = (const geometry::Graph<Dim> &)geometry;
    if (graph.HasPoints() == false) {
        PrintShaderWarning("Binding failed with empty graph.");
        return false;
    }
    copy_pointcloud_functor<Dim> func(graph.HasColors(), option.point_color_option_,
                                      view);
    if (graph.HasNodeColors()) {
        thrust::transform(make_tuple_begin(graph.points_, graph.node_colors_),
                          make_tuple_end(graph.points_, graph.node_colors_),
                          make_tuple_iterator(points, colors), func);
    } else {
        thrust::transform(
                make_tuple_iterator(graph.points_.begin(),
                                    thrust::constant_iterator<Eigen::Vector3f>(
                                            Eigen::Vector3f::Ones())),
                make_tuple_iterator(graph.points_.end(),
                                    thrust::constant_iterator<Eigen::Vector3f>(
                                            Eigen::Vector3f::Ones())),
                make_tuple_iterator(points, colors), func);
    }
    draw_arrays_mode_ = GL_POINTS;
    draw_arrays_size_ = GLsizei(graph.points_.size());
    return true;
}

template <int Dim>
size_t SimpleShaderForGraphNode<Dim>::GetDataSize(
        const geometry::Geometry &geometry) const {
    return ((const geometry::Graph<Dim> &)geometry).points_.size();
}

template class SimpleShaderForGraphNode<2>;
template class SimpleShaderForGraphNode<3>;

template <int Dim>
bool SimpleShaderForGraphEdge<Dim>::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() != geometry::Geometry::GeometryType::Graph) {
        PrintShaderWarning("Rendering type is not geometry::Graph.");
        return false;
    }
    glLineWidth(GLfloat(option.line_width_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

template <int Dim>
bool SimpleShaderForGraphEdge<Dim>::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector4f> &colors) {
    if (geometry.GetGeometryType() != geometry::Geometry::GeometryType::Graph) {
        PrintShaderWarning("Rendering type is not geometry::Graph.");
        return false;
    }
    const geometry::Graph<Dim> &graph = (const geometry::Graph<Dim> &)geometry;
    if (graph.HasLines() == false) {
        PrintShaderWarning("Binding failed with empty geometry::Graph.");
        return false;
    }
    utility::device_vector<thrust::pair<Eigen::Vector3f, Eigen::Vector3f>>
            line_coords(graph.lines_.size());
    line_coordinates_functor<Dim> func_line(
            thrust::raw_pointer_cast(graph.points_.data()));
    thrust::transform(graph.lines_.begin(), graph.lines_.end(),
                      line_coords.begin(), func_line);
    copy_lineset_functor func_cp(thrust::raw_pointer_cast(line_coords.data()),
                                 thrust::raw_pointer_cast(graph.colors_.data()),
                                 graph.HasColors());
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(graph.lines_.size() * 2),
                      make_tuple_iterator(points, colors), func_cp);
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(graph.lines_.size() * 2);
    return true;
}

template <int Dim>
size_t SimpleShaderForGraphEdge<Dim>::GetDataSize(
        const geometry::Geometry &geometry) const {
    return ((const geometry::Graph<Dim> &)geometry).lines_.size() * 2;
}

template class SimpleShaderForGraphEdge<2>;
template class SimpleShaderForGraphEdge<3>;

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
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector4f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::AxisAlignedBoundingBox) {
        PrintShaderWarning(
                "Rendering type is not geometry::AxisAlignedBoundingBox.");
        return false;
    }
    auto lineset = geometry::LineSet<3>::CreateFromAxisAlignedBoundingBox(
            (const geometry::AxisAlignedBoundingBox &)geometry);
    utility::device_vector<thrust::pair<Eigen::Vector3f, Eigen::Vector3f>>
            line_coords(lineset->lines_.size());
    line_coordinates_functor<3> func_line(
            thrust::raw_pointer_cast(lineset->points_.data()));
    thrust::transform(lineset->lines_.begin(), lineset->lines_.end(),
                      line_coords.begin(), func_line);
    copy_lineset_functor func_cp(
            thrust::raw_pointer_cast(line_coords.data()),
            thrust::raw_pointer_cast(lineset->colors_.data()),
            lineset->HasColors());
    thrust::transform(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator(lineset->lines_.size() * 2),
            make_tuple_iterator(points, colors), func_cp);
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(lineset->lines_.size() * 2);
    return true;
}

size_t SimpleShaderForAxisAlignedBoundingBox::GetDataSize(
        const geometry::Geometry &geometry) const {
    auto lineset = geometry::LineSet<3>::CreateFromAxisAlignedBoundingBox(
            (const geometry::AxisAlignedBoundingBox &)geometry);
    return lineset->lines_.size() * 2;
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
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector4f> &colors) {
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

    copy_trianglemesh_functor func(
            thrust::raw_pointer_cast(mesh.vertices_.data()),
            (int *)(thrust::raw_pointer_cast(mesh.triangles_.data())),
            thrust::raw_pointer_cast(mesh.vertex_colors_.data()),
            mesh.HasVertexColors(), option.mesh_color_option_,
            option.default_mesh_color_, view);
    thrust::transform(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator(mesh.triangles_.size() * 3),
            make_tuple_iterator(points, colors), func);
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(mesh.triangles_.size() * 3);
    return true;
}

size_t SimpleShaderForTriangleMesh::GetDataSize(
        const geometry::Geometry &geometry) const {
    return ((const geometry::TriangleMesh &)geometry).triangles_.size() * 3;
}

bool SimpleShaderForVoxelGridLine::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::VoxelGrid) {
        PrintShaderWarning("Rendering type is not geometry::VoxelGrid.");
        return false;
    }
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    return true;
}

bool SimpleShaderForVoxelGridLine::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector4f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::VoxelGrid) {
        PrintShaderWarning("Rendering type is not geometry::VoxelGrid.");
        return false;
    }
    const geometry::VoxelGrid &voxel_grid =
            (const geometry::VoxelGrid &)geometry;
    if (voxel_grid.HasVoxels() == false) {
        PrintShaderWarning("Binding failed with empty voxel grid.");
        return false;
    }

    utility::device_vector<Eigen::Vector3f> vertices(
            voxel_grid.voxels_values_.size() * 8);
    thrust::tiled_range<
            thrust::counting_iterator<size_t>>
            irange(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(8),
                   voxel_grid.voxels_values_.size());
    auto gfunc = geometry::get_grid_index_functor<geometry::Voxel, Eigen::Vector3i>();
    auto begin = thrust::make_transform_iterator(voxel_grid.voxels_values_.begin(), gfunc);
    thrust::repeated_range<decltype(begin)>
            vrange(begin, thrust::make_transform_iterator(voxel_grid.voxels_values_.end(), gfunc), 8);
    geometry::compute_voxel_vertices_functor<Eigen::Vector3i> func1(voxel_grid.origin_, voxel_grid.voxel_size_);
    thrust::transform(make_tuple_begin(irange, vrange), make_tuple_end(irange, vrange),
                      vertices.begin(), func1);

    size_t n_out = voxel_grid.voxels_values_.size() * 12 * 2;
    copy_voxelgrid_line_functor func2(
            thrust::raw_pointer_cast(vertices.data()),
            thrust::raw_pointer_cast(voxel_grid.voxels_values_.data()),
            voxel_grid.HasColors(), option.mesh_color_option_,
            option.default_mesh_color_, view);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_out),
                      make_tuple_iterator(points, colors), func2);
    draw_arrays_mode_ = GL_LINES;
    draw_arrays_size_ = GLsizei(n_out);
    return true;
}

size_t SimpleShaderForVoxelGridLine::GetDataSize(
        const geometry::Geometry &geometry) const {
    return ((const geometry::VoxelGrid &)geometry).voxels_values_.size() * 12 *
           2;
}

bool SimpleShaderForDistanceTransform::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::DistanceTransform) {
        PrintShaderWarning("Rendering type is not geometry::DistanceTransform.");
        return false;
    }
    glPointSize(GLfloat(option.point_size_));
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GLenum(option.GetGLDepthFunc()));
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
}

bool SimpleShaderForDistanceTransform::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector4f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::DistanceTransform) {
        PrintShaderWarning("Rendering type is not geometry::DistanceTransform.");
        return false;
    }
    const geometry::DistanceTransform &dist_trans =
            (const geometry::DistanceTransform &)geometry;
    if (dist_trans.IsEmpty()) {
        PrintShaderWarning("Binding failed with empty distance transform.");
        return false;
    }

    size_t n_out = dist_trans.voxels_.size();
    copy_distance_voxel_functor
            func(dist_trans.voxel_size_, dist_trans.resolution_, dist_trans.origin_,
                 dist_trans.voxel_size_ * dist_trans.resolution_ * 0.1);
    thrust::transform(make_tuple_iterator(thrust::make_counting_iterator<size_t>(0), dist_trans.voxels_.begin()),
                      make_tuple_iterator(thrust::make_counting_iterator(n_out), dist_trans.voxels_.end()),
                      make_tuple_iterator(points, colors), func);
    auto tp_begin = make_tuple_iterator(points, colors);
    thrust::sort(utility::exec_policy(0)->on(0),
                 tp_begin, tp_begin + n_out, alpha_greater_functor());
    draw_arrays_mode_ = GL_POINTS;
    draw_arrays_size_ = GLsizei(n_out);
    return true;
}

size_t SimpleShaderForDistanceTransform::GetDataSize(
        const geometry::Geometry &geometry) const {
    int res = ((const geometry::DistanceTransform &)geometry).resolution_;
    return res * res * res;
}
