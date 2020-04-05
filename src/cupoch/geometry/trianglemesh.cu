#include <thrust/iterator/discard_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/gather.h>

#include "cupoch/geometry/intersection_test.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/helper.h"
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

struct compute_adjacency_list_functor {
    compute_adjacency_list_functor(const Eigen::Vector3i *triangles, Eigen::Vector2i *adjacency_list)
        : triangles_(triangles), adjacency_list_(adjacency_list) {};
    const Eigen::Vector3i *triangles_;
    Eigen::Vector2i *adjacency_list_;
    __device__ void operator()(size_t idx) {
        adjacency_list_[3 * idx] = Eigen::Vector2i(min(triangles_[idx][0], triangles_[idx][1]),
                                                   max(triangles_[idx][0], triangles_[idx][1]));
        adjacency_list_[3 * idx + 1] = Eigen::Vector2i(min(triangles_[idx][1], triangles_[idx][2]),
                                                       max(triangles_[idx][1], triangles_[idx][2]));
        adjacency_list_[3 * idx + 2] = Eigen::Vector2i(min(triangles_[idx][2], triangles_[idx][0]),
                                                       max(triangles_[idx][2], triangles_[idx][0]));
    }
};

struct align_triangle_functor {
    __device__ Eigen::Vector3i operator() (const Eigen::Vector3i& tri) const {
        if (tri(0) <= tri(1)) {
            if (tri(0) <= tri(2)) {
                return Eigen::Vector3i(tri(0), tri(1), tri(2));
            } else {
                return Eigen::Vector3i(tri(2), tri(0), tri(1));
            }
        } else {
            if (tri(1) <= tri(2)) {
                return Eigen::Vector3i(tri(1), tri(2), tri(0));
            } else {
                return Eigen::Vector3i(tri(2), tri(0), tri(1));
            }
        }
    }
};

template <class... Args>
struct check_ref_functor {
    __device__ bool operator()(
            const thrust::tuple<bool, Args...> &x) const {
        const bool ref = thrust::get<0>(x);
        return !ref;
    }
};

struct sample_points_functor {
    sample_points_functor(const Eigen::Vector3f* vertices, const Eigen::Vector3f* vertex_normals,
                          const Eigen::Vector3i* triangles, const Eigen::Vector3f* triangle_normals,
                          const Eigen::Vector3f* vertex_colors, const size_t* n_points_scan,
                          Eigen::Vector3f* points, Eigen::Vector3f* normals, Eigen::Vector3f* colors,
                          bool has_vert_normal, bool use_triangle_normal, bool has_vert_color)
                          : dist_(0.0, 1.0), vertices_(vertices), vertex_normals_(vertex_normals),
                          triangles_(triangles), triangle_normals_(triangle_normals), vertex_colors_(vertex_colors),
                          n_points_scan_(n_points_scan), points_(points), normals_(normals),
                          colors_(colors), has_vert_normal_(has_vert_normal),
                          use_triangle_normal_(use_triangle_normal), has_vert_color_(has_vert_color) {};
    thrust::minstd_rand mt_;
    thrust::uniform_real_distribution<float> dist_;
    const Eigen::Vector3f* vertices_;
    const Eigen::Vector3f* vertex_normals_;
    const Eigen::Vector3i* triangles_;
    const Eigen::Vector3f* triangle_normals_;
    const Eigen::Vector3f* vertex_colors_;
    const size_t* n_points_scan_;
    Eigen::Vector3f* points_;
    Eigen::Vector3f* normals_;
    Eigen::Vector3f* colors_;
    const bool has_vert_normal_;
    const bool use_triangle_normal_;
    const bool has_vert_color_;
    __device__
    void operator() (size_t idx) {
        for (int point_idx = n_points_scan_[idx]; point_idx < n_points_scan_[idx + 1]; ++point_idx) {
            float r1 = dist_(mt_);
            float r2 = dist_(mt_);
            float a = (1 - sqrt(r1));
            float b = sqrt(r1) * (1 - r2);
            float c = sqrt(r1) * r2;

            const Eigen::Vector3i &triangle = triangles_[idx];
            points_[point_idx] = a * vertices_[triangle(0)] +
                                 b * vertices_[triangle(1)] +
                                 c * vertices_[triangle(2)];
            if (has_vert_normal_ && !use_triangle_normal_) {
                normals_[point_idx] = a * vertex_normals_[triangle(0)] +
                                      b * vertex_normals_[triangle(1)] +
                                      c * vertex_normals_[triangle(2)];
            }
            if (use_triangle_normal_) {
                normals_[point_idx] = triangle_normals_[idx];
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

TriangleMesh::TriangleMesh(const geometry::TriangleMesh &other)
    : MeshBase(Geometry::GeometryType::TriangleMesh,
               other.vertices_,
               other.vertex_normals_,
               other.vertex_colors_),
      triangles_(other.triangles_),
      triangle_normals_(other.triangle_normals_),
      adjacency_list_(other.adjacency_list_),
      triangle_uvs_(other.triangle_uvs_) {}

TriangleMesh &TriangleMesh::operator=(const TriangleMesh &other) {
    MeshBase::operator=(other);
    triangles_ = other.triangles_;
    triangle_normals_ = other.triangle_normals_;
    adjacency_list_ = other.adjacency_list_;
    triangle_uvs_ = other.triangle_uvs_;
    return *this;
}

thrust::host_vector<Eigen::Vector3i> TriangleMesh::GetTriangles() const {
    thrust::host_vector<Eigen::Vector3i> triangles = triangles_;
    return triangles;
}

void TriangleMesh::SetTriangles(
        const thrust::host_vector<Eigen::Vector3i> &triangles) {
    triangles_ = triangles;
}

thrust::host_vector<Eigen::Vector3f> TriangleMesh::GetTriangleNormals() const {
    thrust::host_vector<Eigen::Vector3f> triangle_normals = triangle_normals_;
    return triangle_normals;
}

void TriangleMesh::SetTriangleNormals(
        const thrust::host_vector<Eigen::Vector3f> &triangle_normals) {
    triangle_normals_ = triangle_normals;
}

thrust::host_vector<Eigen::Vector2i> TriangleMesh::GetAdjacencyList() const {
    thrust::host_vector<Eigen::Vector2i> adjacency_list = adjacency_list_;
    return adjacency_list;
}

void TriangleMesh::SetAdjacencyList(
        const thrust::host_vector<Eigen::Vector2i> &adjacency_list) {
    adjacency_list_ = adjacency_list;
}

thrust::host_vector<Eigen::Vector2f> TriangleMesh::GetTriangleUVs() const {
    thrust::host_vector<Eigen::Vector2f> triangle_uvs = triangle_uvs_;
    return triangle_uvs;
}

void TriangleMesh::SetTriangleUVs(
        thrust::host_vector<Eigen::Vector2f> &triangle_uvs) {
    triangle_uvs_ = triangle_uvs;
}

TriangleMesh &TriangleMesh::Clear() {
    MeshBase::Clear();
    triangles_.clear();
    triangle_normals_.clear();
    adjacency_list_.clear();
    triangle_uvs_.clear();
    texture_.Clear();
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
    if (HasAdjacencyList()) {
        ComputeAdjacencyList();
    }
    if (HasTriangleUvs() || HasTexture()) {
        utility::LogError(
                "[TriangleMesh] copy of uvs and texture is not implemented "
                "yet");
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
    thrust::sort_by_key(thrust::device, tri_ptr,
                        tri_ptr + copy_tri.size() * 3, nm_thrice.begin());
    auto end = thrust::reduce_by_key(
            thrust::device, tri_ptr, tri_ptr + copy_tri.size() * 3,
            nm_thrice.begin(), thrust::make_discard_iterator(),
            vertex_normals_.begin());
    size_t n_out = thrust::distance(vertex_normals_.begin(), end.second);
    vertex_normals_.resize(n_out);
    if (normalized) {
        NormalizeNormals();
    }
    return *this;
}

TriangleMesh &TriangleMesh::ComputeAdjacencyList() {
    adjacency_list_.clear();
    adjacency_list_.resize(triangles_.size() * 3);
    compute_adjacency_list_functor func(
            thrust::raw_pointer_cast(triangles_.data()),
            thrust::raw_pointer_cast(adjacency_list_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(triangles_.size()), func);
    thrust::sort(adjacency_list_.begin(), adjacency_list_.end());
    auto end = thrust::unique(adjacency_list_.begin(), adjacency_list_.end());
    size_t n_out = thrust::distance(adjacency_list_.begin(), end);
    adjacency_list_.resize(n_out);
    return *this;
}

float TriangleMesh::GetSurfaceArea() const {
    const Eigen::Vector3f* vert_pt = thrust::raw_pointer_cast(vertices_.data());
    const Eigen::Vector3i* tri_pt = thrust::raw_pointer_cast(triangles_.data());
    return thrust::transform_reduce(thrust::make_counting_iterator<size_t>(0),
                                    thrust::make_counting_iterator(triangles_.size()),
                                    [vert_pt, tri_pt] __device__ (size_t idx) -> float { return GetTriangleArea(vert_pt, tri_pt, idx); },
                                    0.0f, thrust::plus<float>());
}

float TriangleMesh::GetSurfaceArea(utility::device_vector<float> &triangle_areas) const {
    const Eigen::Vector3f* vert_pt = thrust::raw_pointer_cast(vertices_.data());
    const Eigen::Vector3i* tri_pt = thrust::raw_pointer_cast(triangles_.data());
    triangle_areas.resize(triangles_.size());
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(triangles_.size()),
                      triangle_areas.begin(),
                      [vert_pt, tri_pt] __device__ (size_t idx) { return GetTriangleArea(vert_pt, tri_pt, idx); });
    return thrust::reduce(triangle_areas.begin(), triangle_areas.end());
}

std::shared_ptr<PointCloud> TriangleMesh::SamplePointsUniformlyImpl(
        size_t number_of_points,
        utility::device_vector<float> &triangle_areas,
        float surface_area,
        bool use_triangle_normal) {
    // triangle areas to cdf
    triangle_areas[0] /= surface_area;
    float* triangle_areas_ptr = thrust::raw_pointer_cast(triangle_areas.data());
    thrust::for_each(thrust::make_counting_iterator<size_t>(1),
                     thrust::make_counting_iterator(triangles_.size()),
                     [triangle_areas_ptr, surface_area] __device__ (size_t idx) {
                         triangle_areas_ptr[idx] = triangle_areas_ptr[idx] / surface_area;
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
    utility::device_vector<size_t> n_points_of_triangle(triangles_.size() + 1, 0);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(triangles_.size()),
                      n_points_of_triangle.begin() + 1,
                      [triangle_areas_ptr, number_of_points] __device__ (size_t idx) {
                          return round(triangle_areas_ptr[idx] * number_of_points);
                      });
    sample_points_functor func(thrust::raw_pointer_cast(vertices_.data()),
                               thrust::raw_pointer_cast(vertex_normals_.data()),
                               thrust::raw_pointer_cast(triangles_.data()),
                               thrust::raw_pointer_cast(triangle_normals_.data()),
                               thrust::raw_pointer_cast(vertex_colors_.data()),
                               thrust::raw_pointer_cast(n_points_of_triangle.data()),
                               thrust::raw_pointer_cast(pcd->points_.data()),
                               thrust::raw_pointer_cast(pcd->normals_.data()),
                               thrust::raw_pointer_cast(pcd->colors_.data()),
                               has_vert_normal, use_triangle_normal, has_vert_color);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(triangles_.size()),
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
    utility::device_vector<int> index_old_to_new(old_vertex_num);
    thrust::sequence(utility::exec_policy(utility::GetStream(0))->on(utility::GetStream(0)),
                     index_new_to_old.begin(), index_new_to_old.end(), 0);
    thrust::sequence(utility::exec_policy(utility::GetStream(1))->on(utility::GetStream(1)),
                     index_old_to_new.begin(), index_old_to_new.end(), 0);
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    size_t k = 0;
    if (has_vert_normal && has_vert_color) {
        thrust::sort_by_key(vertices_.begin(), vertices_.end(),
                            make_tuple_iterator(index_new_to_old.begin(),
                                                vertex_normals_.begin(),
                                                vertex_colors_.begin()));
        auto begin = make_tuple_iterator(index_new_to_old.begin(), vertex_normals_.begin(), vertex_colors_.begin());
        auto end = thrust::unique_by_key(vertices_.begin(), vertices_.end(), begin);
        k = thrust::distance(begin, end.second);
    } else if (has_vert_normal) {
        thrust::sort_by_key(vertices_.begin(), vertices_.end(),
                            make_tuple_iterator(index_new_to_old.begin(),
                                                vertex_normals_.begin()));
        auto begin = make_tuple_iterator(index_new_to_old.begin(), vertex_normals_.begin());
        auto end = thrust::unique_by_key(vertices_.begin(), vertices_.end(), begin);
        k = thrust::distance(begin, end.second);
    } else if (has_vert_color) {
        thrust::sort_by_key(vertices_.begin(), vertices_.end(),
                            make_tuple_iterator(index_new_to_old.begin(),
                                                vertex_colors_.begin()));
        auto begin = make_tuple_iterator(index_new_to_old.begin(), vertex_colors_.begin());
        auto end = thrust::unique_by_key(vertices_.begin(), vertices_.end(), begin);
        k = thrust::distance(begin, end.second);
    } else {
        thrust::sort_by_key(vertices_.begin(), vertices_.end(), index_new_to_old.begin());
        auto end = thrust::unique_by_key(vertices_.begin(), vertices_.end(), index_new_to_old.begin());
        k = thrust::distance(index_new_to_old.begin(), end.second);
    }
    vertices_.resize(k);
    index_new_to_old.resize(k);
    if (has_vert_normal) vertex_normals_.resize(k);
    if (has_vert_color) vertex_colors_.resize(k);
    thrust::gather(index_new_to_old.begin(), index_new_to_old.end(),
                   thrust::make_counting_iterator<size_t>(0), index_old_to_new.begin());
    utility::device_vector<Eigen::Vector3i> new_tri(triangles_.size());
    int* tri_ptr = (int*)thrust::raw_pointer_cast(triangles_.data());
    int* tri_new_ptr = (int*)thrust::raw_pointer_cast(new_tri.data());
    int* index_old_to_new_ptr = thrust::raw_pointer_cast(index_old_to_new.data());
    thrust::transform(thrust::device, tri_ptr, tri_ptr + triangles_.size() * 3, tri_new_ptr,
                      [index_old_to_new_ptr] __device__ (int idx) { return index_old_to_new_ptr[idx]; } );
    triangles_ = new_tri;
    if (HasAdjacencyList()) {
        ComputeAdjacencyList();
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
    thrust::transform(triangles_.begin(), triangles_.end(), new_triangles.begin(), align_triangle_functor());
    if (has_tri_normal) {
        thrust::sort_by_key(new_triangles.begin(), new_triangles.end(), triangle_normals_.begin());
        auto end = thrust::unique_by_key(new_triangles.begin(), new_triangles.end(), triangle_normals_.begin());
        k = thrust::distance(new_triangles.begin(), end.first);
    } else {
        thrust::sort(new_triangles.begin(), new_triangles.end());
        auto end = thrust::unique(new_triangles.begin(), new_triangles.end());
        k = thrust::distance(new_triangles.begin(), end);
    }
    new_triangles.resize(k);
    triangles_ = new_triangles;
    if (has_tri_normal) triangle_normals_.resize(k);
    if (k < old_triangle_num && HasAdjacencyList()) {
        ComputeAdjacencyList();
    }
    utility::LogDebug(
            "[RemoveDuplicatedTriangles] {:d} triangles have been removed.",
            (int)(old_triangle_num - k));

    return *this;
}

TriangleMesh &TriangleMesh::RemoveUnreferencedVertices() {
    utility::device_vector<bool> vertex_has_reference(vertices_.size(), false);
    bool* ref_ptr = thrust::raw_pointer_cast(vertex_has_reference.data());
    int* tri_ptr = (int*)thrust::raw_pointer_cast(triangles_.data());
    thrust::for_each(thrust::device,
                     tri_ptr, tri_ptr + triangles_.size() * 3,
                     [ref_ptr] __device__ (int tri) {
                         ref_ptr[tri] = true;
                     });
    bool has_vert_normal = HasVertexNormals();
    bool has_vert_color = HasVertexColors();
    size_t old_vertex_num = vertices_.size();
    utility::device_vector<int> index_new_to_old(old_vertex_num);
    utility::device_vector<int> index_old_to_new(old_vertex_num);
    thrust::sequence(utility::exec_policy(utility::GetStream(0))->on(utility::GetStream(0)),
                     index_new_to_old.begin(), index_new_to_old.end(), 0);
    thrust::sequence(utility::exec_policy(utility::GetStream(1))->on(utility::GetStream(1)),
                     index_old_to_new.begin(), index_old_to_new.end(), 0);
    size_t k = 0;                                  // new index
    if (!has_vert_normal && !has_vert_color) {
        check_ref_functor<int, Eigen::Vector3f> func;
        auto begin = make_tuple_iterator(vertex_has_reference.begin(), index_new_to_old.begin(),
                                         vertices_.begin());
        auto end = thrust::remove_if(
                begin, make_tuple_iterator(vertex_has_reference.end(), index_new_to_old.end(),
                                           vertices_.end()),
                func);
        k = thrust::distance(begin, end);
    } else if (has_vert_normal && !has_vert_color) {
        check_ref_functor<int, Eigen::Vector3f, Eigen::Vector3f> func;
        auto begin = make_tuple_iterator(vertex_has_reference.begin(), index_new_to_old.begin(),
                                         vertices_.begin(), vertex_normals_.begin());
        auto end = thrust::remove_if(
                begin, make_tuple_iterator(vertex_has_reference.end(), index_new_to_old.end(),
                                           vertices_.end(), vertex_normals_.end()),
                func);
        k = thrust::distance(begin, end);
    } else if (!has_vert_normal && has_vert_color) {
        check_ref_functor<int, Eigen::Vector3f, Eigen::Vector3f> func;
        auto begin = make_tuple_iterator(vertex_has_reference.begin(), index_new_to_old.begin(),
                                         vertices_.begin(), vertex_colors_.begin());
        auto end = thrust::remove_if(
                begin, make_tuple_iterator(vertex_has_reference.end(), index_new_to_old.end(),
                                           vertices_.end(), vertex_colors_.end()),
                func);
        k = thrust::distance(begin, end);
    } else {
        check_ref_functor<int, Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f> func;
        auto begin = make_tuple_iterator(vertex_has_reference.begin(), index_new_to_old.begin(),
                                         vertices_.begin(), vertex_normals_.begin(), vertex_colors_.begin());
        auto end = thrust::remove_if(
                begin, make_tuple_iterator(vertex_has_reference.end(), index_new_to_old.end(),
                                           vertices_.end(), vertex_normals_.end(), vertex_colors_.end()),
                func);
        k = thrust::distance(begin, end);
    }
    vertices_.resize(k);
    if (has_vert_normal) vertex_normals_.resize(k);
    if (has_vert_color) vertex_colors_.resize(k);
    thrust::gather(index_new_to_old.begin(), index_new_to_old.end(),
                   thrust::make_counting_iterator<size_t>(0), index_old_to_new.begin());
    if (k < old_vertex_num) {
        utility::device_vector<Eigen::Vector3i> new_tri(triangles_.size());
        int* tri_ptr = (int*)thrust::raw_pointer_cast(triangles_.data());
        int* tri_new_ptr = (int*)thrust::raw_pointer_cast(new_tri.data());
        int* index_old_to_new_ptr = thrust::raw_pointer_cast(index_old_to_new.data());
        thrust::transform(thrust::device, tri_ptr, tri_ptr + triangles_.size() * 3, tri_new_ptr,
                          [index_old_to_new_ptr] __device__ (int idx) { return index_old_to_new_ptr[idx]; } );
        triangles_ = new_tri;
        if (HasAdjacencyList()) {
            ComputeAdjacencyList();
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
    bool* ref_ptr = thrust::raw_pointer_cast(is_degenerate.data());
    Eigen::Vector3i* tri_ptr = thrust::raw_pointer_cast(triangles_.data());
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(triangles_.size()),
                     [ref_ptr, tri_ptr] __device__ (int i) {
                        if (tri_ptr[i](0) != tri_ptr[i](1) && tri_ptr[i](1) != tri_ptr[i](2) &&
                            tri_ptr[i](2) != tri_ptr[i](0)) {
                            ref_ptr[i] = true;
                        }
                     });
    size_t k = 0;
    if (!has_tri_normal) {
        check_ref_functor<Eigen::Vector3i> func;
        auto begin = make_tuple_iterator(is_degenerate.begin(), triangles_.begin());
        auto end = thrust::remove_if(
                begin, make_tuple_iterator(is_degenerate.end(), triangles_.end()),
                func);
        k = thrust::distance(begin, end);
    } else {
        check_ref_functor<Eigen::Vector3i, Eigen::Vector3f> func;
        auto begin = make_tuple_iterator(is_degenerate.begin(), triangles_.begin(),
                                         triangle_normals_.begin());
        auto end = thrust::remove_if(
                begin, make_tuple_iterator(is_degenerate.end(), triangles_.end(),
                                           triangle_normals_.end()),
                func);
        k = thrust::distance(begin, end);
    }
    triangles_.resize(k);
    if (has_tri_normal) triangle_normals_.resize(k);
    if (k < old_triangle_num && HasAdjacencyList()) {
        ComputeAdjacencyList();
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