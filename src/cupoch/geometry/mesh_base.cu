#include "cupoch/geometry/mesh_base.h"

using namespace cupoch;
using namespace cupoch::geometry;

MeshBase::MeshBase() : Geometry3D(Geometry::GeometryType::MeshBase) {}
MeshBase::~MeshBase() {}

thrust::host_vector<Eigen::Vector3f> MeshBase::GetVertices() const {
    thrust::host_vector<Eigen::Vector3f> vertices = vertices_;
    return vertices;
}

void MeshBase::SetVertices(const thrust::host_vector<Eigen::Vector3f>& vertices) {
    vertices_ = vertices;
}

thrust::host_vector<Eigen::Vector3f> MeshBase::GetVertexNormals() const {
    thrust::host_vector<Eigen::Vector3f> vertex_normals = vertex_normals_;
    return vertex_normals;
}

void MeshBase::SetVertexNormals(const thrust::host_vector<Eigen::Vector3f>& vertex_normals) {
    vertex_normals_ = vertex_normals;
}

thrust::host_vector<Eigen::Vector3f> MeshBase::GetVertexColors() const {
    thrust::host_vector<Eigen::Vector3f> vertex_colors = vertex_colors_;
    return vertex_colors;
}

void MeshBase::SetVertexColors(const thrust::host_vector<Eigen::Vector3f>& vertex_colors) {
    vertex_colors_ = vertex_colors;
}

MeshBase &MeshBase::Clear() {
    vertices_.clear();
    vertex_normals_.clear();
    vertex_colors_.clear();
    return *this;
}

bool MeshBase::IsEmpty() const {return !HasVertices();}

MeshBase::MeshBase(Geometry::GeometryType type) : Geometry3D(type) {}

MeshBase::MeshBase(Geometry::GeometryType type,
         const thrust::device_vector<Eigen::Vector3f> &vertices)
    : Geometry3D(type), vertices_(vertices) {}