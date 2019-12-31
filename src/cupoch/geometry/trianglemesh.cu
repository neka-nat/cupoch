#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/range.h"

#include <Eigen/Geometry>

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct compute_triangle_normals_functor {
    compute_triangle_normals_functor(const Eigen::Vector3f* vertices) : vertices_(vertices) {};
    const Eigen::Vector3f* vertices_;
    __device__
    Eigen::Vector3f operator() (const Eigen::Vector3i& tri) const {
        Eigen::Vector3f v01 = vertices_[tri(1)] - vertices_[tri(0)];
        Eigen::Vector3f v02 = vertices_[tri(2)] - vertices_[tri(0)];
        return v01.cross(v02);
    }
};

struct compute_adjacency_matrix_functor {
    compute_adjacency_matrix_functor(int* adjacency_matrix, size_t n_vertices)
        : adjacency_matrix_(adjacency_matrix), n_vertices_(n_vertices) {};
    int* adjacency_matrix_;
    size_t n_vertices_;
    __device__
    void operator() (const Eigen::Vector3i& triangle) {
        adjacency_matrix_[triangle(0) * n_vertices_ + triangle(1)] = 1;
        adjacency_matrix_[triangle(0) * n_vertices_ + triangle(2)] = 1;
        adjacency_matrix_[triangle(1) * n_vertices_ + triangle(0)] = 1;
        adjacency_matrix_[triangle(1) * n_vertices_ + triangle(2)] = 1;
        adjacency_matrix_[triangle(2) * n_vertices_ + triangle(0)] = 1;
        adjacency_matrix_[triangle(2) * n_vertices_ + triangle(1)] = 1;
    }
};

}

TriangleMesh::TriangleMesh() : MeshBase(Geometry::GeometryType::TriangleMesh) {}
TriangleMesh::~TriangleMesh() {}

TriangleMesh::TriangleMesh(const thrust::device_vector<Eigen::Vector3f> &vertices,
                           const thrust::device_vector<Eigen::Vector3i> &triangles)
    : MeshBase(Geometry::GeometryType::TriangleMesh, vertices), triangles_(triangles) {}

thrust::host_vector<Eigen::Vector3i> TriangleMesh::GetTriangles() const {
    thrust::host_vector<Eigen::Vector3i> triangles = triangles_;
    return triangles;
}

void TriangleMesh::SetTriangles(const thrust::host_vector<Eigen::Vector3i>& triangles) {
    triangles_ = triangles;
}

thrust::host_vector<Eigen::Vector3f> TriangleMesh::GetTriangleNormals() const {
    thrust::host_vector<Eigen::Vector3f> triangle_normals = triangle_normals_;
    return triangle_normals;
}

void TriangleMesh::SetTriangleNormals(const thrust::host_vector<Eigen::Vector3f>& triangle_normals) {
    triangle_normals_ = triangle_normals;
}

thrust::host_vector<int> TriangleMesh::GetAdjacencyMatrix() const {
    thrust::host_vector<int> adjacency_matrix = adjacency_matrix_;
    return adjacency_matrix;
}

void TriangleMesh::SetAdjacencyMatrix(const thrust::host_vector<int>& adjacency_matrix) {
    adjacency_matrix_ = adjacency_matrix;
}

thrust::host_vector<Eigen::Vector2f> TriangleMesh::GetTriangleUVs() const {
    thrust::host_vector<Eigen::Vector2f> triangle_uvs = triangle_uvs_;
    return triangle_uvs;
}

void TriangleMesh::SetTriangleUVs(thrust::host_vector<Eigen::Vector2f>& triangle_uvs) {
    triangle_uvs_ = triangle_uvs;
}

TriangleMesh &TriangleMesh::Clear() {
    MeshBase::Clear();
    triangles_.clear();
    triangle_normals_.clear();
    adjacency_matrix_.clear();
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
        thrust::copy(mesh.triangle_normals_.begin(), mesh.triangle_normals_.end(),
                     triangle_normals_.begin() + old_vert_num);
    } else {
        triangle_normals_.clear();
    }
    size_t n_tri_old = triangles_.size();
    triangles_.resize(triangles_.size() + mesh.triangles_.size());
    Eigen::Vector3i index_shift((int)old_vert_num, (int)old_vert_num,
                                (int)old_vert_num);
    thrust::transform(mesh.triangles_.begin(), mesh.triangles_.end(), triangles_.begin() + n_tri_old,
                      [=] __device__ (const Eigen::Vector3i& tri) {return tri + index_shift;});
    if (HasAdjacencyMatrix()) {
        ComputeAdjacencyMatrix();
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
                     [] __device__ (Eigen::Vector3f& nl) {
                         nl.normalize();
                         if (std::isnan(nl(0))) {
                            nl = Eigen::Vector3f(0.0, 0.0, 1.0);
                        }
                     });
    return *this;
}

TriangleMesh &TriangleMesh::ComputeTriangleNormals(
        bool normalized /* = true*/) {
    triangle_normals_.resize(triangles_.size());
    compute_triangle_normals_functor func(thrust::raw_pointer_cast(vertices_.data()));
    thrust::transform(triangles_.begin(), triangles_.end(), triangle_normals_.begin(), func);
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
    thrust::repeated_range<thrust::device_vector<Eigen::Vector3f>::iterator> range(triangle_normals_.begin(), triangle_normals_.end(), 3);
    thrust::device_vector<Eigen::Vector3f> nm_thrice(triangle_normals_.size() * 3);
    thrust::device_vector<int> key_out(vertices_.size());
    thrust::copy(range.begin(), range.end(), nm_thrice.begin());
    int* tri_ptr = (int*)(thrust::raw_pointer_cast(triangles_.data()));
    thrust::sort_by_key(thrust::device, tri_ptr, tri_ptr + triangles_.size() * 3, nm_thrice.begin());
    thrust::reduce_by_key(thrust::device, tri_ptr, tri_ptr + triangles_.size() * 3, nm_thrice.begin(), key_out.begin(), vertex_normals_.begin());
    if (normalized) {
        NormalizeNormals();
    }
    return *this;
}

TriangleMesh &TriangleMesh::ComputeAdjacencyMatrix() {
    adjacency_matrix_.clear();
    adjacency_matrix_.resize(vertices_.size() * vertices_.size(), 0);
    compute_adjacency_matrix_functor func(thrust::raw_pointer_cast(adjacency_matrix_.data()), vertices_.size());
    thrust::for_each(triangles_.begin(), triangles_.end(), func);
    return *this;
}