#include "cupoch/geometry/meshbase.h"
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/utility/platform.h"

using namespace cupoch;
using namespace cupoch::geometry;

MeshBase::MeshBase() : Geometry3D(Geometry::GeometryType::MeshBase) {}
MeshBase::~MeshBase() {}
MeshBase::MeshBase(const MeshBase& other)
    : Geometry3D(Geometry::GeometryType::MeshBase), vertices_(other.vertices_),
      vertex_normals_(other.vertex_normals_), vertex_colors_(other.vertex_colors_) {}

MeshBase& MeshBase::operator=(const MeshBase& other) {
    vertices_ = other.vertices_;
    vertex_normals_ = other.vertex_normals_;
    vertex_colors_ = other.vertex_colors_;
    return *this;
}

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

Eigen::Vector3f MeshBase::GetMinBound() const {
    return ComputeMinBound(vertices_);
}

Eigen::Vector3f MeshBase::GetMaxBound() const {
    return ComputeMaxBound(vertices_);
}

Eigen::Vector3f MeshBase::GetCenter() const { return ComputeCenter(vertices_); }

AxisAlignedBoundingBox MeshBase::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox::CreateFromPoints(vertices_);
}

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

MeshBase &MeshBase::operator+=(const MeshBase &mesh) {
    if (mesh.IsEmpty()) return (*this);
    size_t old_vert_num = vertices_.size();
    size_t add_vert_num = mesh.vertices_.size();
    size_t new_vert_num = old_vert_num + add_vert_num;
    if ((!HasVertices() || HasVertexNormals()) && mesh.HasVertexNormals()) {
        vertex_normals_.resize(new_vert_num);
        thrust::copy(mesh.vertex_normals_.begin(), mesh.vertex_normals_.end(), vertex_normals_.begin() + old_vert_num);
    } else {
        vertex_normals_.clear();
    }
    if ((!HasVertices() || HasVertexColors()) && mesh.HasVertexColors()) {
        vertex_colors_.resize(new_vert_num);
        thrust::copy(mesh.vertex_colors_.begin(), mesh.vertex_colors_.end(), vertex_colors_.begin() + old_vert_num);
    } else {
        vertex_colors_.clear();
    }
    vertices_.resize(new_vert_num);
    thrust::copy(mesh.vertices_.begin(), mesh.vertices_.end(), vertices_.begin() + old_vert_num);
    return (*this);
}

MeshBase MeshBase::operator+(const MeshBase &mesh) const {
    return (MeshBase(*this) += mesh);
}

MeshBase &MeshBase::NormalizeNormals() {
    thrust::for_each(vertex_normals_.begin(), vertex_normals_.end(),
                     [] __device__ (Eigen::Vector3f& nl) {
                         nl.normalize();
                         if (std::isnan(nl(0))) {
                            nl = Eigen::Vector3f(0.0, 0.0, 1.0);
                        }
                     });
    return *this;
}

MeshBase::MeshBase(Geometry::GeometryType type) : Geometry3D(type) {}

MeshBase::MeshBase(Geometry::GeometryType type,
         const utility::device_vector<Eigen::Vector3f> &vertices)
    : Geometry3D(type), vertices_(vertices) {}

MeshBase::MeshBase(Geometry::GeometryType type,
                   const utility::device_vector<Eigen::Vector3f> &vertices,
                   const utility::device_vector<Eigen::Vector3f> &vertex_normals,
                   const utility::device_vector<Eigen::Vector3f> &vertex_colors)
    : Geometry3D(type), vertices_(vertices),
      vertex_normals_(vertex_normals), vertex_colors_(vertex_colors) {}

MeshBase::MeshBase(Geometry::GeometryType type,
         const thrust::host_vector<Eigen::Vector3f> &vertices)
    : Geometry3D(type), vertices_(vertices) {}
