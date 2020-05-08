#include "cupoch/visualization/shader/phong_shader.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/visualization/shader/shader.h"
#include "cupoch/visualization/utility/color_map.h"
#include <thrust/iterator/constant_iterator.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace cupoch;
using namespace cupoch::visualization;
using namespace cupoch::visualization::glsl;

namespace {

// Coordinates of 8 vertices in a cuboid (assume origin (0,0,0), size 1)
__constant__ int cuboid_vertex_offsets[8][3] = {
    {0, 0, 0}, {1, 0, 0},
    {0, 1, 0}, {1, 1, 0},
    {0, 0, 1}, {1, 0, 1},
    {0, 1, 1}, {1, 1, 1},
};

// Vertex indices of 12 triangles in a cuboid, for right-handed manifold mesh
__constant__ int cuboid_triangles_vertex_indices[12][3] = {
    {0, 2, 1}, {0, 1, 4},
    {0, 4, 2}, {5, 1, 7},
    {5, 7, 4}, {5, 4, 1},
    {3, 7, 1}, {3, 1, 2},
    {3, 2, 7}, {6, 4, 7},
    {6, 7, 2}, {6, 2, 4},
};

__constant__ int cuboid_normals[12][3] = {
    {0, 0, -1}, {0, -1, 0},
    {-1, 0, 0}, {1, 0, 0},
    {0, 0, 1}, {0, -1, 0},
    {1, 0, 0}, {0, 0, -1},
    {0, 1, 0}, {0, 0, 1},
    {0, 1, 0}, {-1, 0, 0},
};

struct copy_pointcloud_functor{
    copy_pointcloud_functor(bool has_colors, RenderOption::PointColorOption color_option, const ViewControl& view)
        : has_colors_(has_colors), color_option_(color_option), view_(view) {};
    const bool has_colors_;
    const RenderOption::PointColorOption color_option_;
    const ViewControl view_;
    const ColorMap::ColorMapOption colormap_option_ = GetGlobalColorMapOption();
    __device__
    thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector4f> operator() (const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>& pt_nm_cl) {
        const Eigen::Vector3f &point = thrust::get<0>(pt_nm_cl);
        const Eigen::Vector3f &normal = thrust::get<1>(pt_nm_cl);
        const Eigen::Vector3f &color = thrust::get<2>(pt_nm_cl);
        Eigen::Vector4f color_tmp;
        color_tmp[3] = 1.0;
        switch (color_option_) {
            case RenderOption::PointColorOption::XCoordinate:
                color_tmp.head<3>() = GetColorMapColor(view_.GetBoundingBox().GetXPercentage(point(0)), colormap_option_);
                break;
            case RenderOption::PointColorOption::YCoordinate:
                color_tmp.head<3>() = GetColorMapColor(view_.GetBoundingBox().GetYPercentage(point(1)), colormap_option_);
                break;
            case RenderOption::PointColorOption::ZCoordinate:
                color_tmp.head<3>() = GetColorMapColor(view_.GetBoundingBox().GetZPercentage(point(2)), colormap_option_);
                break;
            case RenderOption::PointColorOption::Color:
            case RenderOption::PointColorOption::Default:
            default:
                if (has_colors_) {
                    color_tmp.head<3>() = color;
                } else {
                    color_tmp.head<3>() = GetColorMapColor(view_.GetBoundingBox().GetZPercentage(point(2)), colormap_option_);
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
    thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector4f> operator() (size_t k) const {
        int idx = k / 3;
        int vi = triangles_[k];
        const Eigen::Vector3f &vertex = vertices_[vi];

        Eigen::Vector4f color_tmp;
        color_tmp[3] = 1.0;
        switch (color_option_) {
            case RenderOption::MeshColorOption::XCoordinate:
                color_tmp.head<3>() = GetColorMapColor(view_.GetBoundingBox().GetXPercentage(vertex(0)), colormap_option_);
                break;
            case RenderOption::MeshColorOption::YCoordinate:
                color_tmp.head<3>() = GetColorMapColor(view_.GetBoundingBox().GetYPercentage(vertex(1)), colormap_option_);
                break;
            case RenderOption::MeshColorOption::ZCoordinate:
                color_tmp.head<3>() = GetColorMapColor(view_.GetBoundingBox().GetZPercentage(vertex(2)), colormap_option_);
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

        if (shade_option_ ==
            RenderOption::MeshShadeOption::FlatShade) {
            return thrust::make_tuple(vertex, triangle_normals_[idx], color_tmp);
        } else {
            return thrust::make_tuple(vertex, vertex_normals_[vi], color_tmp);
        }
    }

};

template<typename VoxelType>
struct compute_voxel_vertices_functor {
    compute_voxel_vertices_functor(const VoxelType* voxels, const Eigen::Vector3f& origin, float voxel_size)
     : voxels_(voxels), origin_(origin), voxel_size_(voxel_size) {};
    const VoxelType* voxels_;
    const Eigen::Vector3f origin_;
    const float voxel_size_;
    __device__
    Eigen::Vector3f operator() (size_t idx) const {
        int i = idx / 8;
        int j = idx % 8;
        const VoxelType &voxel = voxels_[i];
        // 8 vertices in a voxel
        Eigen::Vector3f base_vertex =
                origin_ + voxel.grid_index_.template cast<float>() * voxel_size_;
        const auto offset_v = Eigen::Vector3f(cuboid_vertex_offsets[j][0],
                                              cuboid_vertex_offsets[j][1],
                                              cuboid_vertex_offsets[j][2]);
        return base_vertex + offset_v * voxel_size_;
    }
};

struct default_color_functor {
    __host__ __device__ default_color_functor() {};
    __host__ __device__ ~default_color_functor() {};
    __host__ __device__ default_color_functor(const default_color_functor& other) {};
    __device__ Eigen::Vector3f color(const geometry::Voxel& voxel) const {
        return voxel.color_;
    }
    __device__ float alpha(const geometry::Voxel& voxel) const {
        return 1.0;
    }
};

struct occupancy_color_functor {
    __host__ __device__ occupancy_color_functor(float occ_prob_thres_log)
     : occ_prob_thres_log_(occ_prob_thres_log) {};
    const float occ_prob_thres_log_;
    __host__ __device__ ~occupancy_color_functor() {};
    __host__ __device__ occupancy_color_functor(const occupancy_color_functor& other)
     : occ_prob_thres_log_(other.occ_prob_thres_log_) {};
    __device__ Eigen::Vector3f color(const geometry::Voxel& voxel) const {
        geometry::OccupancyVoxel ocv = (const geometry::OccupancyVoxel &)voxel;
        return (ocv.prob_log_ > occ_prob_thres_log_) ? ocv.color_ : Eigen::Vector3f(0.0, 1.0, 0.0);
    }
    __device__ float alpha(const geometry::Voxel& voxel) const {
        geometry::OccupancyVoxel ocv = (const geometry::OccupancyVoxel &)voxel;
        return (ocv.prob_log_ > occ_prob_thres_log_) ? 1.0 : 0.2;
    }
};

template<typename VoxelType, typename ColorFuncType>
struct copy_voxelgrid_face_functor {
    copy_voxelgrid_face_functor(const Eigen::Vector3f* vertices, const VoxelType* voxels, bool has_colors,
                                RenderOption::MeshColorOption color_option,
                                const Eigen::Vector3f& default_mesh_color,
                                const ViewControl& view, const ColorFuncType& cfunc)
                                : vertices_(vertices), voxels_(voxels), has_colors_(has_colors),
                                 color_option_(color_option), default_mesh_color_(default_mesh_color),
                                 view_(view), cfunc_(cfunc) {};
    const Eigen::Vector3f* vertices_;
    const VoxelType* voxels_;
    const bool has_colors_;
    const RenderOption::MeshColorOption color_option_;
    const Eigen::Vector3f default_mesh_color_;
    const ViewControl view_;
    const ColorFuncType cfunc_;
    const ColorMap::ColorMapOption colormap_option_ = GetGlobalColorMapOption();
    __device__
    thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector4f> operator() (size_t idx) const {
        int i = idx / (12 * 3);
        int jk = idx % (12 * 3);
        int j = jk / 3;
        int k = jk % 3;
        // Voxel color (applied to all points)
        Eigen::Vector4f voxel_color;
        voxel_color[3] = cfunc_.alpha(voxels_[i]);
        switch (color_option_) {
            case RenderOption::MeshColorOption::XCoordinate:
                voxel_color.head<3>() = GetColorMapColor(view_.GetBoundingBox().GetXPercentage(vertices_[i * 8](0)), colormap_option_);
                break;
            case RenderOption::MeshColorOption::YCoordinate:
                voxel_color.head<3>() = GetColorMapColor(view_.GetBoundingBox().GetYPercentage(vertices_[i * 8](1)), colormap_option_);
                break;
            case RenderOption::MeshColorOption::ZCoordinate:
                voxel_color.head<3>() = GetColorMapColor(view_.GetBoundingBox().GetZPercentage(vertices_[i * 8](2)), colormap_option_);
                break;
            case RenderOption::MeshColorOption::Color:
                if (has_colors_) {
                    voxel_color.head<3>() = cfunc_.color(voxels_[i]);
                    break;
                }
            case RenderOption::MeshColorOption::Default:
            default:
                voxel_color.head<3>() = default_mesh_color_;
                break;
        }
        return thrust::make_tuple(vertices_[i * 8 + cuboid_triangles_vertex_indices[j][k]],
            Eigen::Vector3f(cuboid_normals[j][0], cuboid_normals[j][1], cuboid_normals[j][2]), voxel_color);
    }
};

struct alpha_greater_functor {
    __device__ bool operator () (const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector4f>& lhs,
                                 const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector4f>& rhs) {
        const Eigen::Vector4f& lc = thrust::get<2>(lhs);
        const Eigen::Vector4f& rc = thrust::get<2>(rhs);
        return lc[3] > rc[3];
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
    UnbindGeometry(true);
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
    glBufferData(GL_ARRAY_BUFFER, num_data_size * sizeof(Eigen::Vector4f), 0, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[2], vertex_color_buffer_, cudaGraphicsMapFlagsNone));

    Eigen::Vector3f* raw_points_ptr;
    Eigen::Vector3f* raw_normals_ptr;
    Eigen::Vector4f* raw_colors_ptr;
    size_t n_bytes;
    cudaSafeCall(cudaGraphicsMapResources(3, cuda_graphics_resources_));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_points_ptr, &n_bytes, cuda_graphics_resources_[0]));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_normals_ptr, &n_bytes, cuda_graphics_resources_[1]));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_colors_ptr, &n_bytes, cuda_graphics_resources_[2]));
    thrust::device_ptr<Eigen::Vector3f> dev_points_ptr = thrust::device_pointer_cast(raw_points_ptr);
    thrust::device_ptr<Eigen::Vector3f> dev_normals_ptr = thrust::device_pointer_cast(raw_normals_ptr);
    thrust::device_ptr<Eigen::Vector4f> dev_colors_ptr = thrust::device_pointer_cast(raw_colors_ptr);

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
    glVertexAttribPointer(vertex_color_, 4, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_normal_);
    glDisableVertexAttribArray(vertex_color_);
    return true;
}

void PhongShader::UnbindGeometry(bool finalize) {
    if (bound_) {
        if (!finalize) {
            cudaSafeCall(cudaGraphicsUnregisterResource(cuda_graphics_resources_[0]));
            cudaSafeCall(cudaGraphicsUnregisterResource(cuda_graphics_resources_[1]));
            cudaSafeCall(cudaGraphicsUnregisterResource(cuda_graphics_resources_[2]));
        }
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

bool PhongShaderForVoxelGridFace::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::VoxelGrid) {
        PrintShaderWarning("Rendering type is not geometry::VoxelGrid.");
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
    SetLighting(view, option);
    return true;
}

bool PhongShaderForVoxelGridFace::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector3f> &normals,
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

    utility::device_vector<Eigen::Vector3f> vertices(voxel_grid.voxels_values_.size() * 8);
    compute_voxel_vertices_functor<geometry::Voxel> func1(thrust::raw_pointer_cast(voxel_grid.voxels_values_.data()),
                                                          voxel_grid.origin_, voxel_grid.voxel_size_);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(voxel_grid.voxels_values_.size() * 8),
                      vertices.begin(), func1);

    size_t n_out = voxel_grid.voxels_values_.size() * 12 * 3;
    copy_voxelgrid_face_functor<geometry::Voxel, default_color_functor> func2(
        thrust::raw_pointer_cast(vertices.data()),
        thrust::raw_pointer_cast(voxel_grid.voxels_values_.data()),
        voxel_grid.HasColors(), option.mesh_color_option_,
        option.default_mesh_color_,
        view, default_color_functor());
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_out),
                      make_tuple_iterator(points, normals, colors), func2);
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(n_out);

    return true;
}

size_t PhongShaderForVoxelGridFace::GetDataSize(const geometry::Geometry &geometry) const {
    return ((const geometry::VoxelGrid &)geometry).voxels_keys_.size() * 12 * 3;
}

bool PhongShaderForOccupancyGrid::PrepareRendering(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::OccupancyGrid) {
        PrintShaderWarning("Rendering type is not geometry::OccupancyGrid.");
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
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
}

bool PhongShaderForOccupancyGrid::PrepareBinding(
        const geometry::Geometry &geometry,
        const RenderOption &option,
        const ViewControl &view,
        thrust::device_ptr<Eigen::Vector3f> &points,
        thrust::device_ptr<Eigen::Vector3f> &normals,
        thrust::device_ptr<Eigen::Vector4f> &colors) {
    if (geometry.GetGeometryType() !=
        geometry::Geometry::GeometryType::OccupancyGrid) {
        PrintShaderWarning("Rendering type is not geometry::OccupancyGrid.");
        return false;
    }
    const geometry::OccupancyGrid &occupancy_grid =
            (const geometry::OccupancyGrid &)geometry;
    if (occupancy_grid.HasVoxels() == false) {
        PrintShaderWarning("Binding failed with empty voxel grid.");
        return false;
    }

    utility::device_vector<geometry::OccupancyVoxel> voxels =
            (occupancy_grid.visualize_free_area_) ? occupancy_grid.ExtractKnownVoxels() : occupancy_grid.ExtractOccupiedVoxels();
    utility::device_vector<Eigen::Vector3f> vertices(voxels.size() * 8);
    Eigen::Vector3f origin = occupancy_grid.origin_ - 0.5 * occupancy_grid.voxel_size_ * Eigen::Vector3f::Constant(occupancy_grid.resolution_);
    compute_voxel_vertices_functor<geometry::OccupancyVoxel> func1(thrust::raw_pointer_cast(voxels.data()),
                                                                   origin, occupancy_grid.voxel_size_);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(voxels.size() * 8),
                      vertices.begin(), func1);

    size_t n_out = voxels.size() * 12 * 3;
    copy_voxelgrid_face_functor<geometry::OccupancyVoxel, occupancy_color_functor> func2(
        thrust::raw_pointer_cast(vertices.data()),
        thrust::raw_pointer_cast(voxels.data()),
        occupancy_grid.HasColors(), option.mesh_color_option_,
        option.default_mesh_color_, view,
        occupancy_color_functor(occupancy_grid.occ_prob_thres_log_));
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_out),
                      make_tuple_iterator(points, normals, colors), func2);
    auto begin = make_tuple_iterator(points, normals, colors);
    thrust::sort(begin, begin + n_out, alpha_greater_functor());
    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = GLsizei(n_out);

    return true;
}

size_t PhongShaderForOccupancyGrid::GetDataSize(const geometry::Geometry &geometry) const {
    const geometry::OccupancyGrid &occupancy_grid = (const geometry::OccupancyGrid &)geometry;
    return (occupancy_grid.visualize_free_area_) ? occupancy_grid.CountKnownVoxels() * 12 * 3 : occupancy_grid.CountOccupiedVoxels() * 12 * 3;
}