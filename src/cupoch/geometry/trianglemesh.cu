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
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/random.h>

#include "cupoch/geometry/geometry_functor.h"
#include "cupoch/geometry/intersection_test.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/helper.h"
#include "cupoch/utility/platform.h"
#include "cupoch/utility/range.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct compute_triangle_normals_functor {
    compute_triangle_normals_functor(const Eigen::Vector3f *vertices)
        : vertices_(vertices){};
    const Eigen::Vector3f *vertices_;
    __device__ Eigen::Vector3f operator()(const Eigen::Vector3i &tri) const {
        Eigen::Vector3f v01 = vertices_[tri(1)] - vertices_[tri(0)];
        Eigen::Vector3f v02 = vertices_[tri(2)] - vertices_[tri(0)];
        return v01.cross(v02);
    }
};

struct compute_edge_list_functor {
    compute_edge_list_functor(const Eigen::Vector3i *triangles,
                              Eigen::Vector2i *edge_list)
        : triangles_(triangles), edge_list_(edge_list){};
    const Eigen::Vector3i *triangles_;
    Eigen::Vector2i *edge_list_;
    __device__ void operator()(size_t idx) {
        const int min01 = min(triangles_[idx][0], triangles_[idx][1]);
        const int max01 = max(triangles_[idx][0], triangles_[idx][1]);
        edge_list_[6 * idx] = Eigen::Vector2i(min01, max01);
        edge_list_[6 * idx + 1] = Eigen::Vector2i(max01, min01);
        const int min12 = min(triangles_[idx][1], triangles_[idx][2]);
        const int max12 = max(triangles_[idx][1], triangles_[idx][2]);
        edge_list_[6 * idx + 2] = Eigen::Vector2i(min12, max12);
        edge_list_[6 * idx + 3] = Eigen::Vector2i(max12, min12);
        const int min20 = min(triangles_[idx][2], triangles_[idx][0]);
        const int max20 = max(triangles_[idx][2], triangles_[idx][0]);
        edge_list_[6 * idx + 4] = Eigen::Vector2i(min20, max20);
        edge_list_[6 * idx + 5] = Eigen::Vector2i(max20, min20);
    }
};

struct compute_old_to_new_index_functor {
    compute_old_to_new_index_functor(const int *idx_offsets,
                                     const int *index_new_to_old,
                                     int *index_old_to_new)
        : idx_offsets_(idx_offsets),
          index_new_to_old_(index_new_to_old),
          index_old_to_new_(index_old_to_new){};
    const int *idx_offsets_;
    const int *index_new_to_old_;
    int *index_old_to_new_;
    __device__ void operator()(size_t idx) {
        int si = idx_offsets_[idx];
        int ei = idx_offsets_[idx + 1];
        for (int i = si; i < ei; ++i) {
            index_old_to_new_[index_new_to_old_[i]] = idx;
        }
    }
};

struct align_triangle_functor {
    __device__ Eigen::Vector3i operator()(const Eigen::Vector3i &tri) const {
        if (tri(0) <= tri(1)) {
            return (tri(0) <= tri(2)) ? Eigen::Vector3i(tri(0), tri(1), tri(2))
                                      : Eigen::Vector3i(tri(2), tri(0), tri(1));
        } else {
            return (tri(1) <= tri(2)) ? Eigen::Vector3i(tri(1), tri(2), tri(0))
                                      : Eigen::Vector3i(tri(2), tri(0), tri(1));
        }
    }
};

struct edge_first_eq_functor {
    __device__ bool operator()(const Eigen::Vector2i &lhs,
                               const Eigen::Vector2i &rhs) {
        return lhs[0] == rhs[0];
    };
};

struct weighted_vec_functor {
    __device__ Eigen::Vector3f operator()(
            const thrust::tuple<Eigen::Vector3f, float> &x) {
        return thrust::get<1>(x) * thrust::get<0>(x);
    };
};

struct compute_weights_from_edges_functor {
    compute_weights_from_edges_functor(const Eigen::Vector3f *vertices)
        : vertices_(vertices){};
    const Eigen::Vector3f *vertices_;
    __device__ float operator()(const Eigen::Vector2i &edge) {
        const auto dist = (vertices_[edge[0]] - vertices_[edge[1]]).norm();
        return 1. / (dist + 1e-12);
    }
};

void FilterSmoothLaplacianHelper(
        std::shared_ptr<TriangleMesh> &mesh,
        utility::device_vector<Eigen::Vector3f> &prev_vertices,
        utility::device_vector<Eigen::Vector3f> &prev_vertex_normals,
        utility::device_vector<Eigen::Vector3f> &prev_vertex_colors,
        utility::device_vector<Eigen::Vector3f> &vertex_sums,
        utility::device_vector<Eigen::Vector3f> &normal_sums,
        utility::device_vector<Eigen::Vector3f> &color_sums,
        utility::device_vector<float> &weights,
        utility::device_vector<float> &total_weights,
        float lambda,
        bool filter_vertex,
        bool filter_normal,
        bool filter_color) {
    typedef utility::device_vector<Eigen::Vector3f>::iterator ElementIterator;
    auto filter_fn =
            [lambda] __device__(
                    const thrust::tuple<Eigen::Vector3f, float, Eigen::Vector3f>
                            &x) -> Eigen::Vector3f {
        const Eigen::Vector3f &prv = thrust::get<0>(x);
        const Eigen::Vector3f &sum = thrust::get<2>(x);
        return prv + lambda * (sum / thrust::get<1>(x) - prv);
    };
    auto runs = [&mesh, &filter_fn, &weights, &total_weights] (auto& prev, auto& sums, auto &res) {
        auto tritr = thrust::make_transform_iterator(
                mesh->edge_list_.begin(), element_get_functor<Eigen::Vector2i, 1>());
        thrust::permutation_iterator<ElementIterator, decltype(tritr)> pmitr(
                prev.begin(), tritr);
        thrust::reduce_by_key(
                utility::exec_policy(0), mesh->edge_list_.begin(),
                mesh->edge_list_.end(),
                thrust::make_transform_iterator(
                        make_tuple_iterator(pmitr, weights.begin()),
                        weighted_vec_functor()),
                thrust::make_discard_iterator(), sums.begin(),
                edge_first_eq_functor());
        thrust::transform(
                make_tuple_begin(prev, total_weights, sums),
                make_tuple_end(prev, total_weights, sums),
                res.begin(), filter_fn);
    };
    if (filter_vertex) {
        runs(prev_vertices, vertex_sums, mesh->vertices_);
    }
    if (filter_normal) {
        runs(prev_vertex_normals, normal_sums, mesh->vertex_normals_);
    }
    if (filter_color) {
        runs(prev_vertex_colors, color_sums, mesh->vertex_colors_);
    }
}

template <class... Args>
struct check_ref_functor {
    __device__ bool operator()(const thrust::tuple<Args...> &x) const {
        const bool ref = thrust::get<0>(x);
        return !ref;
    }
};

struct sample_points_functor {
    sample_points_functor(const Eigen::Vector3f *vertices,
                          const Eigen::Vector3f *vertex_normals,
                          const Eigen::Vector3i *triangles,
                          const Eigen::Vector3f *triangle_normals,
                          const Eigen::Vector3f *vertex_colors,
                          const size_t *n_points_scan,
                          Eigen::Vector3f *points,
                          Eigen::Vector3f *normals,
                          Eigen::Vector3f *colors,
                          bool has_vert_normal,
                          bool use_triangle_normal,
                          bool has_vert_color,
                          int n_pallarel)
        : vertices_(vertices),
          vertex_normals_(vertex_normals),
          triangles_(triangles),
          triangle_normals_(triangle_normals),
          vertex_colors_(vertex_colors),
          n_points_scan_(n_points_scan),
          points_(points),
          normals_(normals),
          colors_(colors),
          has_vert_normal_(has_vert_normal),
          use_triangle_normal_(use_triangle_normal),
          has_vert_color_(has_vert_color),
          n_pallarel_(n_pallarel){};
    const Eigen::Vector3f *vertices_;
    const Eigen::Vector3f *vertex_normals_;
    const Eigen::Vector3i *triangles_;
    const Eigen::Vector3f *triangle_normals_;
    const Eigen::Vector3f *vertex_colors_;
    const size_t *n_points_scan_;
    Eigen::Vector3f *points_;
    Eigen::Vector3f *normals_;
    Eigen::Vector3f *colors_;
    const bool has_vert_normal_;
    const bool use_triangle_normal_;
    const bool has_vert_color_;
    const int n_pallarel_;
    __device__ void operator()(size_t idx) {
        int i = idx / n_pallarel_;
        int j = idx % n_pallarel_;
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(0, 1.0);
        rng.discard(idx);
        const Eigen::Vector3i triangle = triangles_[i];
        const Eigen::Vector3f v1 = vertices_[triangle(0)];
        const Eigen::Vector3f v2 = vertices_[triangle(1)];
        const Eigen::Vector3f v3 = vertices_[triangle(2)];
        for (int point_idx = n_points_scan_[i] + j;
             point_idx < n_points_scan_[i + 1]; point_idx += j + 1) {
            float r1 = dist(rng);
            float r2 = dist(rng);
            float a = (1 - sqrtf(r1));
            float b = sqrtf(r1) * (1 - r2);
            float c = sqrtf(r1) * r2;

            points_[point_idx] = a * v1 + b * v2 + c * v3;
            if (has_vert_normal_ && !use_triangle_normal_) {
                normals_[point_idx] = a * vertex_normals_[triangle(0)] +
                                      b * vertex_normals_[triangle(1)] +
                                      c * vertex_normals_[triangle(2)];
            } else if (use_triangle_normal_) {
                normals_[point_idx] = triangle_normals_[i];
            }
            if (has_vert_color_) {
                colors_[point_idx] = a * vertex_colors_[triangle(0)] +
                                     b * vertex_colors_[triangle(1)] +
                                     c * vertex_colors_[triangle(2)];
            }
        }
    }
};

struct check_self_intersecting_triangles {
    check_self_intersecting_triangles(const Eigen::Vector3i *triangles,
                                      const Eigen::Vector3f *vertices,
                                      int n_triangles)
        : triangles_(triangles),
          vertices_(vertices),
          n_triangles_(n_triangles){};
    const Eigen::Vector3i *triangles_;
    const Eigen::Vector3f *vertices_;
    const int n_triangles_;
    __device__ Eigen::Vector2i operator()(size_t idx) const {
        int tidx0 = idx / n_triangles_;
        int tidx1 = idx % n_triangles_;
        if (tidx0 >= tidx1 || tidx0 == n_triangles_ - 1) {
            return Eigen::Vector2i(-1, -1);
        }
        const Eigen::Vector3i &tria_p = triangles_[tidx0];
        const Eigen::Vector3f &p0 = vertices_[tria_p(0)];
        const Eigen::Vector3f &p1 = vertices_[tria_p(1)];
        const Eigen::Vector3f &p2 = vertices_[tria_p(2)];
        const Eigen::Vector3i &tria_q = triangles_[tidx1];
        // check if neighbour triangle
        if (tria_p(0) == tria_q(0) || tria_p(0) == tria_q(1) ||
            tria_p(0) == tria_q(2) || tria_p(1) == tria_q(0) ||
            tria_p(1) == tria_q(1) || tria_p(1) == tria_q(2) ||
            tria_p(2) == tria_q(0) || tria_p(2) == tria_q(1) ||
            tria_p(2) == tria_q(2)) {
            return Eigen::Vector2i(-1, -1);
        }
        // check for intersection
        const Eigen::Vector3f &q0 = vertices_[tria_q(0)];
        const Eigen::Vector3f &q1 = vertices_[tria_q(1)];
        const Eigen::Vector3f &q2 = vertices_[tria_q(2)];
        if (intersection_test::TriangleTriangle3d(p0, p1, p2, q0, q1, q2)) {
            return Eigen::Vector2i(tidx0, tidx1);
        }
        return Eigen::Vector2i(-1, -1);
    }
};

}  // namespace

TriangleMesh::TriangleMesh() : MeshBase(Geometry::GeometryType::TriangleMesh) {}
TriangleMesh::~TriangleMesh() {}

TriangleMesh::TriangleMesh(
        const utility::device_vector<Eigen::Vector3f> &vertices,
        const utility::device_vector<Eigen::Vector3i> &triangles)
    : MeshBase(Geometry::GeometryType::TriangleMesh, vertices),
      triangles_(triangles) {}

TriangleMesh::TriangleMesh(
        const thrust::host_vector<Eigen::Vector3f> &vertices,
        const thrust::host_vector<Eigen::Vector3i> &triangles)
    : MeshBase(Geometry::GeometryType::TriangleMesh, vertices),
      triangles_(triangles) {}

TriangleMesh::TriangleMesh(const std::vector<Eigen::Vector3f> &vertices,
                           const std::vector<Eigen::Vector3i> &triangles)
    : MeshBase(Geometry::GeometryType::TriangleMesh, vertices),
      triangles_(triangles) {}

TriangleMesh::TriangleMesh(const geometry::TriangleMesh &other)
    : MeshBase(Geometry::GeometryType::TriangleMesh,
               other.vertices_,
               other.vertex_normals_,
               other.vertex_colors_),
      triangles_(other.triangles_),
      triangle_normals_(other.triangle_normals_),
      edge_list_(other.edge_list_),
      triangle_uvs_(other.triangle_uvs_),
      texture_(other.texture_) {}

TriangleMesh &TriangleMesh::operator=(const TriangleMesh &other) {
    MeshBase::operator=(other);
    triangles_ = other.triangles_;
    triangle_normals_ = other.triangle_normals_;
    edge_list_ = other.edge_list_;
    triangle_uvs_ = other.triangle_uvs_;
    texture_ = other.texture_;
    return *this;
}

std::vector<Eigen::Vector3i> TriangleMesh::GetTriangles() const {
    std::vector<Eigen::Vector3i> triangles(triangles_.size());
    copy_device_to_host(triangles_, triangles);
    return triangles;
}

void TriangleMesh::SetTriangles(
        const thrust::host_vector<Eigen::Vector3i> &triangles) {
    triangles_ = triangles;
}

void TriangleMesh::SetTriangles(const std::vector<Eigen::Vector3i> &triangles) {
    triangles_.resize(triangles.size());
    copy_host_to_device(triangles, triangles_);
}

std::vector<Eigen::Vector3f> TriangleMesh::GetTriangleNormals() const {
    std::vector<Eigen::Vector3f> triangle_normals(triangle_normals_.size());
    copy_device_to_host(triangle_normals_, triangle_normals);
    return triangle_normals;
}

void TriangleMesh::SetTriangleNormals(
        const thrust::host_vector<Eigen::Vector3f> &triangle_normals) {
    triangle_normals_ = triangle_normals;
}

void TriangleMesh::SetTriangleNormals(const std::vector<Eigen::Vector3f> &triangle_normals) {
    triangle_normals_.resize(triangle_normals.size());
    copy_host_to_device(triangle_normals, triangle_normals_);
}

std::vector<Eigen::Vector2i> TriangleMesh::GetEdgeList() const {
    std::vector<Eigen::Vector2i> edge_list(edge_list_.size());
    copy_device_to_host(edge_list_, edge_list);
    return edge_list;
}

void TriangleMesh::SetEdgeList(
        const thrust::host_vector<Eigen::Vector2i> &edge_list) {
    edge_list_ = edge_list;
}

void TriangleMesh::SetEdgeList(const std::vector<Eigen::Vector2i> &edge_list) {
    edge_list_.resize(edge_list.size());
    copy_host_to_device(edge_list, edge_list_);
}

std::vector<Eigen::Vector2f> TriangleMesh::GetTriangleUVs() const {
    std::vector<Eigen::Vector2f> triangle_uvs(triangle_uvs_.size());
    copy_device_to_host(triangle_uvs_, triangle_uvs);
    return triangle_uvs;
}

void TriangleMesh::SetTriangleUVs(
        const thrust::host_vector<Eigen::Vector2f> &triangle_uvs) {
    triangle_uvs_ = triangle_uvs;
}

void TriangleMesh::SetTriangleUVs(const std::vector<Eigen::Vector2f> &triangle_uvs) {
    triangle_uvs_.resize(triangle_uvs.size());
    copy_host_to_device(triangle_uvs, triangle_uvs_);
}

TriangleMesh &TriangleMesh::Clear() {
    MeshBase::Clear();
    triangles_.clear();
    triangle_normals_.clear();
    edge_list_.clear();
    triangle_uvs_.clear();
    texture_.Clear();
    return *this;
}

TriangleMesh &TriangleMesh::Transform(const Eigen::Matrix4f &transformation) {
    MeshBase::Transform(transformation);
    TransformNormals(transformation, triangle_normals_);
    return *this;
}

TriangleMesh &TriangleMesh::Rotate(const Eigen::Matrix3f &R, bool center) {
    MeshBase::Rotate(R, center);
    RotateNormals(R, triangle_normals_);
    return *this;
}

TriangleMesh &TriangleMesh::operator+=(const TriangleMesh &mesh) {
    if (mesh.IsEmpty()) return (*this);
    size_t old_vert_num = vertices_.size();
    MeshBase::operator+=(mesh);
    size_t old_tri_num = triangles_.size();
    size_t add_tri_num = mesh.triangles_.size();
    size_t new_tri_num = old_tri_num + add_tri_num;
    if ((!HasTriangles() || HasTriangleNormals()) &&
        mesh.HasTriangleNormals()) {
        triangle_normals_.resize(new_tri_num);
        thrust::copy(mesh.triangle_normals_.begin(),
                     mesh.triangle_normals_.end(),
                     triangle_normals_.begin() + old_tri_num);
    } else {
        triangle_normals_.clear();
    }
    size_t n_tri_old = triangles_.size();
    triangles_.resize(triangles_.size() + mesh.triangles_.size());
    Eigen::Vector3i index_shift((int)old_vert_num, (int)old_vert_num,
                                (int)old_vert_num);
    thrust::transform(mesh.triangles_.begin(), mesh.triangles_.end(),
                      triangles_.begin() + n_tri_old,
                      [=] __device__(const Eigen::Vector3i &tri) {
                          return tri + index_shift;
                      });
    if (HasEdgeList()) {
        ComputeEdgeList();
    }
    if (HasTriangleUvs() || HasTexture()) {
        utility::LogError(
                "[TriangleMesh] copy of uvs and texture is not implemented "
                "yet");
    } else if (mesh.HasTriangleUvs() || mesh.HasTexture()) {
        triangle_uvs_ = mesh.triangle_uvs_;
        texture_ = mesh.texture_;
    }
    return (*this);
}

TriangleMesh TriangleMesh::operator+(const TriangleMesh &mesh) const {
    return (TriangleMesh(*this) += mesh);
}

TriangleMesh &TriangleMesh::NormalizeNormals() {
    MeshBase::NormalizeNormals();
    thrust::for_each(triangle_normals_.begin(), triangle_normals_.end(),
                     [] __device__(Eigen::Vector3f & nl) {
                         nl.normalize();
                         if (isnan(nl(0))) {
                             nl = Eigen::Vector3f(0.0, 0.0, 1.0);
                         }
                     });
    return *this;
}

TriangleMesh &TriangleMesh::ComputeTriangleNormals(
        bool normalized /* = true*/) {
    triangle_normals_.resize(triangles_.size());
    compute_triangle_normals_functor func(
            thrust::raw_pointer_cast(vertices_.data()));
    thrust::transform(triangles_.begin(), triangles_.end(),
                      triangle_normals_.begin(), func);
    if (normalized) {
        NormalizeNormals();
    }
    return *this;
}

TriangleMesh &TriangleMesh::ComputeVertexNormals(bool normalized /* = true*/) {
    if (HasTriangleNormals() == false) {
        ComputeTriangleNormals(false);
    }
    vertex_normals_.resize(vertices_.size());
    thrust::repeated_range<utility::device_vector<Eigen::Vector3f>::iterator>
            range(triangle_normals_.begin(), triangle_normals_.end(), 3);
    utility::device_vector<Eigen::Vector3f> nm_thrice(triangle_normals_.size() *
                                                      3);
    thrust::copy(range.begin(), range.end(), nm_thrice.begin());
    utility::device_vector<Eigen::Vector3i> copy_tri = triangles_;
    int *tri_ptr = (int *)(thrust::raw_pointer_cast(copy_tri.data()));
    thrust::sort_by_key(utility::exec_policy(0), tri_ptr,
                        tri_ptr + copy_tri.size() * 3, nm_thrice.begin());
    auto end = thrust::reduce_by_key(
            utility::exec_policy(0), tri_ptr,
            tri_ptr + copy_tri.size() * 3, nm_thrice.begin(),
            thrust::make_discard_iterator(), vertex_normals_.begin());
    size_t n_out = thrust::distance(vertex_normals_.begin(), end.second);
    vertex_normals_.resize(n_out);
    if (normalized) {
        NormalizeNormals();
    }
    return *this;
}

TriangleMesh &TriangleMesh::ComputeEdgeList() {
    edge_list_.clear();
    edge_list_.resize(triangles_.size() * 6);
    compute_edge_list_functor func(thrust::raw_pointer_cast(triangles_.data()),
                                   thrust::raw_pointer_cast(edge_list_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(triangles_.size()), func);
    thrust::sort(utility::exec_policy(0), edge_list_.begin(),
                 edge_list_.end());
    auto end = thrust::unique(utility::exec_policy(0),
                              edge_list_.begin(), edge_list_.end());
    size_t n_out = thrust::distance(edge_list_.begin(), end);
    edge_list_.resize(n_out);
    return *this;
}

std::shared_ptr<TriangleMesh> TriangleMesh::FilterSharpen(
        int number_of_iterations, float strength, FilterScope scope) const {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            HasVertexColors();

    utility::device_vector<Eigen::Vector3f> prev_vertices = vertices_;
    utility::device_vector<Eigen::Vector3f> prev_vertex_normals =
            vertex_normals_;
    utility::device_vector<Eigen::Vector3f> prev_vertex_colors = vertex_colors_;

    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_.resize(vertices_.size());
    mesh->vertex_normals_.resize(vertex_normals_.size());
    mesh->vertex_colors_.resize(vertex_colors_.size());
    mesh->triangles_ = triangles_;
    mesh->edge_list_ = edge_list_;
    if (!mesh->HasEdgeList()) {
        mesh->ComputeEdgeList();
    }

    utility::device_vector<Eigen::Vector3f> vertex_sums(vertices_.size());
    utility::device_vector<Eigen::Vector3f> normal_sums(vertex_normals_.size());
    utility::device_vector<Eigen::Vector3f> color_sums(vertex_colors_.size());
    utility::device_vector<int> counts(vertices_.size());
    typedef utility::device_vector<Eigen::Vector3f>::iterator ElementIterator;
    auto filter_fn =
            [strength] __device__(
                    const thrust::tuple<Eigen::Vector3f, int, Eigen::Vector3f>
                            &x) -> Eigen::Vector3f {
        const Eigen::Vector3f &prv = thrust::get<0>(x);
        const Eigen::Vector3f &sum = thrust::get<2>(x);
        return prv + strength * (prv * thrust::get<1>(x) - sum);
    };
    thrust::reduce_by_key(utility::exec_policy(0),
                          mesh->edge_list_.begin(), mesh->edge_list_.end(),
                          thrust::make_constant_iterator(1),
                          thrust::make_discard_iterator(), counts.begin(),
                          edge_first_eq_functor());
    auto runs = [&mesh, &filter_fn, &counts] (auto& prev, auto& sums, auto& res) {
        auto tritr = thrust::make_transform_iterator(
                mesh->edge_list_.begin(),
                element_get_functor<Eigen::Vector2i, 1>());
        thrust::permutation_iterator<ElementIterator, decltype(tritr)>
                pmitr(prev.begin(), tritr);
        thrust::reduce_by_key(utility::exec_policy(0),
                              mesh->edge_list_.begin(),
                              mesh->edge_list_.end(), pmitr,
                              thrust::make_discard_iterator(),
                              sums.begin(), edge_first_eq_functor());
        thrust::transform(
                make_tuple_begin(prev, counts, sums),
                make_tuple_end(prev, counts, sums),
                res.begin(), filter_fn);
    };
    for (int iter = 0; iter < number_of_iterations; ++iter) {
        if (filter_vertex) {
            runs(prev_vertices, vertex_sums, mesh->vertices_);
        }
        if (filter_normal) {
            runs(prev_vertex_normals, normal_sums, mesh->vertex_normals_);
        }
        if (filter_color) {
            runs(prev_vertex_colors, color_sums, mesh->vertex_colors_);
        }
        if (iter < number_of_iterations - 1) {
            thrust::swap(mesh->vertices_, prev_vertices);
            thrust::swap(mesh->vertex_normals_, prev_vertex_normals);
            thrust::swap(mesh->vertex_colors_, prev_vertex_colors);
        }
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::FilterSmoothSimple(
        int number_of_iterations, FilterScope scope) const {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            HasVertexColors();

    utility::device_vector<Eigen::Vector3f> prev_vertices = vertices_;
    utility::device_vector<Eigen::Vector3f> prev_vertex_normals =
            vertex_normals_;
    utility::device_vector<Eigen::Vector3f> prev_vertex_colors = vertex_colors_;

    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_.resize(vertices_.size());
    mesh->vertex_normals_.resize(vertex_normals_.size());
    mesh->vertex_colors_.resize(vertex_colors_.size());
    mesh->triangles_ = triangles_;
    mesh->edge_list_ = edge_list_;
    if (!mesh->HasEdgeList()) {
        mesh->ComputeEdgeList();
    }

    utility::device_vector<Eigen::Vector3f> vertex_sums(vertices_.size());
    utility::device_vector<Eigen::Vector3f> normal_sums(vertex_normals_.size());
    utility::device_vector<Eigen::Vector3f> color_sums(vertex_colors_.size());
    utility::device_vector<int> counts(vertices_.size());
    typedef utility::device_vector<Eigen::Vector3f>::iterator ElementIterator;
    auto filter_fn =
            [] __device__(
                    const thrust::tuple<Eigen::Vector3f, int, Eigen::Vector3f>
                            &x) -> Eigen::Vector3f {
        const Eigen::Vector3f &prv = thrust::get<0>(x);
        const Eigen::Vector3f &sum = thrust::get<2>(x);
        return (prv + sum) / (1.0 + thrust::get<1>(x));
    };
    thrust::reduce_by_key(utility::exec_policy(0),
                          mesh->edge_list_.begin(), mesh->edge_list_.end(),
                          thrust::make_constant_iterator(1),
                          thrust::make_discard_iterator(), counts.begin(),
                          edge_first_eq_functor());
    auto runs = [&mesh, &counts, &filter_fn] (auto& prev, auto& sums, auto& res) {
        auto tritr = thrust::make_transform_iterator(
                mesh->edge_list_.begin(),
                element_get_functor<Eigen::Vector2i, 1>());
        thrust::permutation_iterator<ElementIterator, decltype(tritr)>
                pmitr(prev.begin(), tritr);
        thrust::reduce_by_key(utility::exec_policy(0),
                              mesh->edge_list_.begin(),
                              mesh->edge_list_.end(), pmitr,
                              thrust::make_discard_iterator(),
                              sums.begin(), edge_first_eq_functor());
        thrust::transform(
                make_tuple_begin(prev, counts, sums),
                make_tuple_end(prev, counts, sums),
                res.begin(), filter_fn);
    };
    for (int iter = 0; iter < number_of_iterations; ++iter) {
        if (filter_vertex) {
            runs(prev_vertices, vertex_sums, mesh->vertices_);
        }
        if (filter_normal) {
            runs(prev_vertex_normals, normal_sums, mesh->vertex_normals_);
        }
        if (filter_color) {
            runs(prev_vertex_colors, color_sums, mesh->vertex_colors_);
        }
        if (iter < number_of_iterations - 1) {
            thrust::swap(mesh->vertices_, prev_vertices);
            thrust::swap(mesh->vertex_normals_, prev_vertex_normals);
            thrust::swap(mesh->vertex_colors_, prev_vertex_colors);
        }
    }

    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::FilterSmoothLaplacian(
        int number_of_iterations, float lambda, FilterScope scope) const {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            HasVertexColors();

    utility::device_vector<Eigen::Vector3f> prev_vertices = vertices_;
    utility::device_vector<Eigen::Vector3f> prev_vertex_normals =
            vertex_normals_;
    utility::device_vector<Eigen::Vector3f> prev_vertex_colors = vertex_colors_;

    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_.resize(vertices_.size());
    mesh->vertex_normals_.resize(vertex_normals_.size());
    mesh->vertex_colors_.resize(vertex_colors_.size());
    mesh->triangles_ = triangles_;
    mesh->edge_list_ = edge_list_;
    if (!mesh->HasEdgeList()) {
        mesh->ComputeEdgeList();
    }

    utility::device_vector<Eigen::Vector3f> vertex_sums(mesh->vertices_.size());
    utility::device_vector<Eigen::Vector3f> normal_sums(
            mesh->vertex_normals_.size());
    utility::device_vector<Eigen::Vector3f> color_sums(
            mesh->vertex_colors_.size());
    utility::device_vector<float> weights(mesh->edge_list_.size());
    utility::device_vector<float> total_weights(mesh->vertices_.size());
    compute_weights_from_edges_functor func(
            thrust::raw_pointer_cast(prev_vertices.data()));
    thrust::transform(mesh->edge_list_.begin(), mesh->edge_list_.end(),
                      weights.begin(), func);
    thrust::reduce_by_key(utility::exec_policy(0),
                          mesh->edge_list_.begin(), mesh->edge_list_.end(),
                          weights.begin(), thrust::make_discard_iterator(),
                          total_weights.begin(), edge_first_eq_functor());

    for (int iter = 0; iter < number_of_iterations; ++iter) {
        FilterSmoothLaplacianHelper(
                mesh, prev_vertices, prev_vertex_normals, prev_vertex_colors,
                vertex_sums, normal_sums, color_sums, weights, total_weights,
                lambda, filter_vertex, filter_normal, filter_color);
        if (iter < number_of_iterations - 1) {
            std::swap(mesh->vertices_, prev_vertices);
            std::swap(mesh->vertex_normals_, prev_vertex_normals);
            std::swap(mesh->vertex_colors_, prev_vertex_colors);
        }
    }
    return mesh;
}

std::shared_ptr<TriangleMesh> TriangleMesh::FilterSmoothTaubin(
        int number_of_iterations,
        float lambda,
        float mu,
        FilterScope scope) const {
    bool filter_vertex =
            scope == FilterScope::All || scope == FilterScope::Vertex;
    bool filter_normal =
            (scope == FilterScope::All || scope == FilterScope::Normal) &&
            HasVertexNormals();
    bool filter_color =
            (scope == FilterScope::All || scope == FilterScope::Color) &&
            HasVertexColors();

    utility::device_vector<Eigen::Vector3f> prev_vertices = vertices_;
    utility::device_vector<Eigen::Vector3f> prev_vertex_normals =
            vertex_normals_;
    utility::device_vector<Eigen::Vector3f> prev_vertex_colors = vertex_colors_;

    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>();
    mesh->vertices_.resize(vertices_.size());
    mesh->vertex_normals_.resize(vertex_normals_.size());
    mesh->vertex_colors_.resize(vertex_colors_.size());
    mesh->triangles_ = triangles_;
    mesh->edge_list_ = edge_list_;
    if (!mesh->HasEdgeList()) {
        mesh->ComputeEdgeList();
    }

    utility::device_vector<Eigen::Vector3f> vertex_sums(mesh->vertices_.size());
    utility::device_vector<Eigen::Vector3f> normal_sums(
            mesh->vertex_normals_.size());
    utility::device_vector<Eigen::Vector3f> color_sums(
            mesh->vertex_colors_.size());
    utility::device_vector<float> weights(mesh->edge_list_.size());
    utility::device_vector<float> total_weights(mesh->vertices_.size());
    compute_weights_from_edges_functor func(
            thrust::raw_pointer_cast(prev_vertices.data()));
    thrust::transform(mesh->edge_list_.begin(), mesh->edge_list_.end(),
                      weights.begin(), func);
    thrust::reduce_by_key(utility::exec_policy(0),
                          mesh->edge_list_.begin(), mesh->edge_list_.end(),
                          weights.begin(), thrust::make_discard_iterator(),
                          total_weights.begin(), edge_first_eq_functor());
    for (int iter = 0; iter < number_of_iterations; ++iter) {
        FilterSmoothLaplacianHelper(
                mesh, prev_vertices, prev_vertex_normals, prev_vertex_colors,
                vertex_sums, normal_sums, color_sums, weights, total_weights,
                lambda, filter_vertex, filter_normal, filter_color);
        thrust::swap(mesh->vertices_, prev_vertices);
        thrust::swap(mesh->vertex_normals_, prev_vertex_normals);
        thrust::swap(mesh->vertex_colors_, prev_vertex_colors);
        FilterSmoothLaplacianHelper(
                mesh, prev_vertices, prev_vertex_normals, prev_vertex_colors,
                vertex_sums, normal_sums, color_sums, weights, total_weights,
                mu, filter_vertex, filter_normal, filter_color);
        if (iter < number_of_iterations - 1) {
            thrust::swap(mesh->vertices_, prev_vertices);
            thrust::swap(mesh->vertex_normals_, prev_vertex_normals);
            thrust::swap(mesh->vertex_colors_, prev_vertex_colors);
        }
    }
    return mesh;
}

float TriangleMesh::GetSurfaceArea() const {
    const Eigen::Vector3f *vert_pt = thrust::raw_pointer_cast(vertices_.data());
    const Eigen::Vector3i *tri_pt = thrust::raw_pointer_cast(triangles_.data());
    return thrust::transform_reduce(
            utility::exec_policy(0),
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator(triangles_.size()),
            [vert_pt, tri_pt] __device__(size_t idx) -> float {
                return GetTriangleArea(vert_pt, tri_pt, idx);
            },
            0.0f, thrust::plus<float>());
}

float TriangleMesh::GetSurfaceArea(
        utility::device_vector<float> &triangle_areas) const {
    const Eigen::Vector3f *vert_pt = thrust::raw_pointer_cast(vertices_.data());
    const Eigen::Vector3i *tri_pt = thrust::raw_pointer_cast(triangles_.data());
    triangle_areas.resize(triangles_.size());
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(triangles_.size()),
                      triangle_areas.begin(),
                      [vert_pt, tri_pt] __device__(size_t idx) {
                          return GetTriangleArea(vert_pt, tri_pt, idx);
                      });
    return thrust::reduce(utility::exec_policy(0),
                          triangle_areas.begin(), triangle_areas.end());
}

std::shared_ptr<PointCloud> TriangleMesh::SamplePointsUniformlyImpl(
        size_t number_of_points,
        utility::device_vector<float> &triangle_areas,
        float surface_area,
        bool use_triangle_normal) {
    // triangle areas to cdf
    thrust::for_each(triangle_areas.begin(), triangle_areas.end(),
                     [surface_area] __device__(float &triangle_area) {
                         triangle_area /= surface_area;
                     });
    thrust::inclusive_scan(triangle_areas.begin(), triangle_areas.end(),
                           triangle_areas.begin());

    // sample point cloud
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    auto pcd = std::make_shared<PointCloud>();
    pcd->points_.resize(number_of_points);
    if (has_vert_normal || use_triangle_normal) {
        pcd->normals_.resize(number_of_points);
    }
    if (use_triangle_normal && !HasTriangleNormals()) {
        ComputeTriangleNormals(true);
    }
    if (has_vert_color) {
        pcd->colors_.resize(number_of_points);
    }
    utility::device_vector<size_t> n_points_of_triangle(triangles_.size() + 1,
                                                        0);
    thrust::transform(
            triangle_areas.begin(), triangle_areas.end(),
            n_points_of_triangle.begin() + 1,
            [number_of_points] __device__(float triangle_area) {
                return (size_t)llrintf(triangle_area * number_of_points);
            });
    int n_pallarel = number_of_points / triangles_.size();
    sample_points_functor func(
            thrust::raw_pointer_cast(vertices_.data()),
            thrust::raw_pointer_cast(vertex_normals_.data()),
            thrust::raw_pointer_cast(triangles_.data()),
            thrust::raw_pointer_cast(triangle_normals_.data()),
            thrust::raw_pointer_cast(vertex_colors_.data()),
            thrust::raw_pointer_cast(n_points_of_triangle.data()),
            thrust::raw_pointer_cast(pcd->points_.data()),
            thrust::raw_pointer_cast(pcd->normals_.data()),
            thrust::raw_pointer_cast(pcd->colors_.data()), has_vert_normal,
            use_triangle_normal, has_vert_color, n_pallarel);
    thrust::for_each(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator(triangles_.size() * n_pallarel),
            func);
    return pcd;
}

std::shared_ptr<PointCloud> TriangleMesh::SamplePointsUniformly(
        size_t number_of_points, bool use_triangle_normal /* = false */) {
    if (number_of_points <= 0) {
        utility::LogError("[SamplePointsUniformly] number_of_points <= 0");
    }
    if (triangles_.size() == 0) {
        utility::LogError(
                "[SamplePointsUniformly] input mesh has no triangles");
        throw std::runtime_error("input mesh has no triangles");
    }

    // Compute area of each triangle and sum surface area
    utility::device_vector<float> triangle_areas;
    float surface_area = GetSurfaceArea(triangle_areas);

    return SamplePointsUniformlyImpl(number_of_points, triangle_areas,
                                     surface_area, use_triangle_normal);
}

TriangleMesh &TriangleMesh::RemoveDuplicatedVertices() {
    size_t old_vertex_num = vertices_.size();
    utility::device_vector<int> index_new_to_old(old_vertex_num);
    thrust::sequence(index_new_to_old.begin(), index_new_to_old.end());
    utility::device_vector<int> idx_offsets(old_vertex_num, 0);
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    size_t k = 0;
    auto runs = [&vertices = vertices_, &index_new_to_old, &idx_offsets] (auto&... params) -> size_t {
        thrust::sort_by_key(utility::exec_policy(0), vertices.begin(),
                            vertices.end(),
                            make_tuple_begin(index_new_to_old, params...));
        auto end0 = thrust::reduce_by_key(
                utility::exec_policy(0), vertices.begin(),
                vertices.end(), thrust::make_constant_iterator<int>(1),
                thrust::make_discard_iterator(), idx_offsets.begin());
        idx_offsets.resize(thrust::distance(idx_offsets.begin(), end0.second) +
                           1);
        thrust::exclusive_scan(idx_offsets.begin(), idx_offsets.end(),
                               idx_offsets.begin());
        auto begin = make_tuple_begin(params...);
        auto end1 = thrust::unique_by_key(utility::exec_policy(0),
                                          vertices.begin(), vertices.end(),
                                          begin);
        return thrust::distance(vertices.begin(), end1.first);
    };
    if (has_vert_normal && has_vert_color) {
        k = runs(vertex_normals_, vertex_colors_);
    } else if (has_vert_normal) {
        thrust::discard_iterable dummy;
        k = runs(vertex_normals_, dummy);
    } else if (has_vert_color) {
        thrust::discard_iterable dummy;
        k = runs(vertex_colors_, dummy);
    } else {
        thrust::discard_iterable dummy;
        k = runs(dummy, dummy);
    }
    vertices_.resize(k);
    if (has_vert_normal) vertex_normals_.resize(k);
    if (has_vert_color) vertex_colors_.resize(k);
    utility::device_vector<int> index_old_to_new(old_vertex_num);
    compute_old_to_new_index_functor func(
            thrust::raw_pointer_cast(idx_offsets.data()),
            thrust::raw_pointer_cast(index_new_to_old.data()),
            thrust::raw_pointer_cast(index_old_to_new.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(k), func);
    utility::device_vector<Eigen::Vector3i> new_tri(triangles_.size());
    int *tri_ptr = (int *)thrust::raw_pointer_cast(triangles_.data());
    int *tri_new_ptr = (int *)thrust::raw_pointer_cast(new_tri.data());
    int *index_old_to_new_ptr =
            thrust::raw_pointer_cast(index_old_to_new.data());
    thrust::transform(thrust::device, tri_ptr, tri_ptr + triangles_.size() * 3,
                      tri_new_ptr, [index_old_to_new_ptr] __device__(int idx) {
                          return index_old_to_new_ptr[idx];
                      });
    triangles_ = new_tri;
    if (HasEdgeList()) {
        ComputeEdgeList();
    }
    utility::LogDebug(
            "[RemoveDuplicatedVertices] {:d} vertices have been removed.",
            (int)(old_vertex_num - k));

    return *this;
}

TriangleMesh &TriangleMesh::RemoveDuplicatedTriangles() {
    if (HasTriangleUvs()) {
        utility::LogWarning(
                "[RemoveDuplicatedTriangles] This mesh contains triangle uvs "
                "that are not handled in this function");
    }
    bool has_tri_normal = HasTriangleNormals();
    size_t old_triangle_num = triangles_.size();
    size_t k = 0;
    utility::device_vector<Eigen::Vector3i> new_triangles(old_triangle_num);
    thrust::transform(triangles_.begin(), triangles_.end(),
                      new_triangles.begin(), align_triangle_functor());
    if (has_tri_normal) {
        thrust::sort_by_key(utility::exec_policy(0),
                            new_triangles.begin(), new_triangles.end(),
                            triangle_normals_.begin());
        auto end = thrust::unique_by_key(
                utility::exec_policy(0), new_triangles.begin(),
                new_triangles.end(), triangle_normals_.begin());
        k = thrust::distance(new_triangles.begin(), end.first);
    } else {
        thrust::sort(utility::exec_policy(0), new_triangles.begin(),
                     new_triangles.end());
        auto end = thrust::unique(utility::exec_policy(0),
                                  new_triangles.begin(), new_triangles.end());
        k = thrust::distance(new_triangles.begin(), end);
    }
    new_triangles.resize(k);
    thrust::swap(triangles_, new_triangles);
    if (has_tri_normal) triangle_normals_.resize(k);
    if (k < old_triangle_num && HasEdgeList()) {
        ComputeEdgeList();
    }
    utility::LogDebug(
            "[RemoveDuplicatedTriangles] {:d} triangles have been removed.",
            (int)(old_triangle_num - k));

    return *this;
}

TriangleMesh &TriangleMesh::RemoveUnreferencedVertices() {
    utility::device_vector<bool> vertex_has_reference(vertices_.size(), false);
    bool *ref_ptr = thrust::raw_pointer_cast(vertex_has_reference.data());
    int *tri_ptr = (int *)thrust::raw_pointer_cast(triangles_.data());
    thrust::for_each(thrust::device, tri_ptr, tri_ptr + triangles_.size() * 3,
                     [ref_ptr] __device__(int tri) { ref_ptr[tri] = true; });
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    size_t old_vertex_num = vertices_.size();
    utility::device_vector<int> index_new_to_old(old_vertex_num);
    thrust::sequence(index_new_to_old.begin(), index_new_to_old.end(), 0);
    size_t k = 0;
    if (!has_vert_normal && !has_vert_color) {
        k = remove_if_vectors_without_resize(
                utility::exec_policy(0),
                check_ref_functor<bool, int, Eigen::Vector3f>(),
                vertex_has_reference, index_new_to_old, vertices_);
    } else if (has_vert_normal && !has_vert_color) {
        k = remove_if_vectors_without_resize(
                utility::exec_policy(0),
                check_ref_functor<bool, int, Eigen::Vector3f,
                                  Eigen::Vector3f>(),
                vertex_has_reference, index_new_to_old, vertices_,
                vertex_normals_);
    } else if (!has_vert_normal && has_vert_color) {
        k = remove_if_vectors_without_resize(
                utility::exec_policy(0),
                check_ref_functor<bool, int, Eigen::Vector3f,
                                  Eigen::Vector3f>(),
                vertex_has_reference, index_new_to_old, vertices_,
                vertex_colors_);
    } else {
        k = remove_if_vectors_without_resize(
                utility::exec_policy(0),
                check_ref_functor<bool, int, Eigen::Vector3f, Eigen::Vector3f,
                                  Eigen::Vector3f>(),
                vertex_has_reference, index_new_to_old, vertices_,
                vertex_normals_, vertex_colors_);
    }
    vertices_.resize(k);
    if (has_vert_normal) vertex_normals_.resize(k);
    if (has_vert_color) vertex_colors_.resize(k);
    if (k < old_vertex_num) {
        thrust::fill(index_new_to_old.begin() + k, index_new_to_old.end(),
                     old_vertex_num);
        utility::device_vector<int> index_old_to_new(old_vertex_num + 1, -1);
        thrust::scatter(thrust::make_counting_iterator<size_t>(0),
                        thrust::make_counting_iterator(old_vertex_num),
                        index_new_to_old.begin(), index_old_to_new.begin());
        utility::device_vector<Eigen::Vector3i> new_tri(triangles_.size());
        int *tri_ptr = (int *)thrust::raw_pointer_cast(triangles_.data());
        int *tri_new_ptr = (int *)thrust::raw_pointer_cast(new_tri.data());
        int *index_old_to_new_ptr =
                thrust::raw_pointer_cast(index_old_to_new.data());
        thrust::transform(thrust::device, tri_ptr,
                          tri_ptr + triangles_.size() * 3, tri_new_ptr,
                          [index_old_to_new_ptr] __device__(int idx) {
                              return index_old_to_new_ptr[idx];
                          });
        thrust::swap(triangles_, new_tri);
        if (HasEdgeList()) {
            ComputeEdgeList();
        }
    }
    utility::LogDebug(
            "[RemoveUnreferencedVertices] {:d} vertices have been removed.",
            (int)(old_vertex_num - k));

    return *this;
}

TriangleMesh &TriangleMesh::RemoveDegenerateTriangles() {
    if (HasTriangleUvs()) {
        utility::LogWarning(
                "[RemoveDegenerateTriangles] This mesh contains triangle uvs "
                "that are not handled in this function");
    }
    bool has_tri_normal = HasTriangleNormals();
    size_t old_triangle_num = triangles_.size();
    utility::device_vector<bool> is_degenerate(triangles_.size(), false);
    bool *ref_ptr = thrust::raw_pointer_cast(is_degenerate.data());
    Eigen::Vector3i *tri_ptr = thrust::raw_pointer_cast(triangles_.data());
    thrust::for_each(thrust::device, thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(triangles_.size()),
                     [ref_ptr, tri_ptr] __device__(int i) {
                         if (tri_ptr[i](0) != tri_ptr[i](1) &&
                             tri_ptr[i](1) != tri_ptr[i](2) &&
                             tri_ptr[i](2) != tri_ptr[i](0)) {
                             ref_ptr[i] = true;
                         }
                     });
    if (!has_tri_normal) {
        remove_if_vectors(utility::exec_policy(0),
                          check_ref_functor<bool, Eigen::Vector3i>(),
                          is_degenerate, triangles_);
    } else {
        remove_if_vectors(
                utility::exec_policy(0),
                check_ref_functor<bool, Eigen::Vector3i, Eigen::Vector3f>(),
                is_degenerate, triangles_, triangle_normals_);
    }
    size_t k = triangles_.size();
    if (k < old_triangle_num && HasEdgeList()) {
        ComputeEdgeList();
    }
    utility::LogDebug(
            "[RemoveDegenerateTriangles] {:d} triangles have been "
            "removed.",
            (int)(old_triangle_num - k));
    return *this;
}

utility::device_vector<Eigen::Vector2i>
TriangleMesh::GetSelfIntersectingTriangles() const {
    const size_t n_triangles2 = triangles_.size() * triangles_.size();
    utility::device_vector<Eigen::Vector2i> self_intersecting_triangles(
            n_triangles2);
    check_self_intersecting_triangles func(
            thrust::raw_pointer_cast(triangles_.data()),
            thrust::raw_pointer_cast(vertices_.data()), triangles_.size());
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_triangles2),
                      self_intersecting_triangles.begin(), func);
    auto end = thrust::remove_if(
            self_intersecting_triangles.begin(),
            self_intersecting_triangles.end(),
            [] __device__(const Eigen::Vector2i &idxs) { return idxs[0] < 0; });
    self_intersecting_triangles.resize(
            thrust::distance(self_intersecting_triangles.begin(), end));
    return self_intersecting_triangles;
}