#include "cupoch/geometry/trianglemesh.h"

using namespace cupoch;
using namespace cupoch::geometry;

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

thrust::host_vector<int> TriangleMesh::GetAdjacencyList() const {
    thrust::host_vector<int> adjacency_list = adjacency_list_;
    return adjacency_list;
}

void TriangleMesh::SetAdjacencyList(const thrust::host_vector<int>& adjacency_list) {
    adjacency_list_ = adjacency_list;
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
    adjacency_list_.clear();
    triangle_uvs_.clear();
    texture_.Clear();
    return *this;
}