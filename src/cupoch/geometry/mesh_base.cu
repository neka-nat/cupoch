#include "cupoch/geometry/mesh_base.h"
#include "cupoch/utility/platform.h"

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

MeshBase &MeshBase::Transform(const Eigen::Matrix4f &transformation) {
    TransformPoints(utility::GetStream(0), transformation, vertices_);
    TransformNormals(utility::GetStream(1), transformation, vertex_normals_);
    cudaDeviceSynchronize();
    return *this;
}

MeshBase &MeshBase::Translate(const Eigen::Vector3f &translation,
                              bool relative) {
    TranslatePoints(translation, vertices_, relative);
    return *this;
}

MeshBase &MeshBase::Scale(const float scale, bool center) {
    ScalePoints(scale, vertices_, center);
    return *this;
}

MeshBase &MeshBase::Rotate(const Eigen::Matrix3f &R, bool center) {
    RotatePoints(utility::GetStream(0), R, vertices_, center);
    RotateNormals(utility::GetStream(0), R, vertex_normals_);
    cudaDeviceSynchronize();
    return *this;
}

MeshBase::MeshBase(Geometry::GeometryType type) : Geometry3D(type) {}

MeshBase::MeshBase(Geometry::GeometryType type,
         const thrust::device_vector<Eigen::Vector3f> &vertices)
    : Geometry3D(type), vertices_(vertices) {}