#pragma once

#include <Eigen/Core>
#include "cupoch/utility/device_vector.h"
#include <thrust/host_vector.h>

#include "cupoch/geometry/geometry3d.h"
#include "cupoch/utility/helper.h"

namespace cupoch {
namespace geometry {

class PointCloud;
class TriangleMesh;

class MeshBase : public Geometry3D {
public:
    /// Indicates the method that is used for mesh simplification if multiple
    /// vertices are combined to a single one.
    /// \param Average indicates that the average position is computed as
    /// output.
    /// \param Quadric indicates that the distance to the adjacent triangle
    /// planes is minimized. Cf. "Simplifying Surfaces with Color and Texture
    /// using Quadric Error Metrics" by Garland and Heckbert.
    enum class SimplificationContraction { Average, Quadric };

    /// Indicates the scope of filter operations.
    /// \param All indicates that all properties (color, normal,
    /// vertex position) are filtered.
    /// \param Color indicates that only the colors are filtered.
    /// \param Normal indicates that only the normals are filtered.
    /// \param Vertex indicates that only the vertex positions are filtered.
    enum class FilterScope { All, Color, Normal, Vertex };

    MeshBase();
    virtual ~MeshBase();
    MeshBase(const MeshBase& other);
    MeshBase& operator=(const MeshBase& other);

    thrust::host_vector<Eigen::Vector3f> GetVertices() const;
    void SetVertices(const thrust::host_vector<Eigen::Vector3f>& vertices);

    thrust::host_vector<Eigen::Vector3f> GetVertexNormals() const;
    void SetVertexNormals(const thrust::host_vector<Eigen::Vector3f>& vertex_normals);

    thrust::host_vector<Eigen::Vector3f> GetVertexColors() const;
    void SetVertexColors(const thrust::host_vector<Eigen::Vector3f>& vertex_colors);

public:
    virtual MeshBase &Clear() override;
    virtual bool IsEmpty() const override;
    virtual Eigen::Vector3f GetMinBound() const override;
    virtual Eigen::Vector3f GetMaxBound() const override;
    virtual Eigen::Vector3f GetCenter() const override;
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    virtual MeshBase &Transform(const Eigen::Matrix4f &transformation) override;
    virtual MeshBase &Translate(const Eigen::Vector3f &translation,
                                bool relative = true) override;
    virtual MeshBase &Scale(const float scale, bool center = true) override;
    virtual MeshBase &Rotate(const Eigen::Matrix3f &R,
                             bool center = true) override;

    MeshBase &operator+=(const MeshBase &mesh);
    MeshBase operator+(const MeshBase &mesh) const;

    __host__ __device__
    bool HasVertices() const { return vertices_.size() > 0; }

    __host__ __device__
    bool HasVertexNormals() const {
        return vertices_.size() > 0 &&
               vertex_normals_.size() == vertices_.size();
    }

    __host__ __device__
    bool HasVertexColors() const {
        return vertices_.size() > 0 &&
               vertex_colors_.size() == vertices_.size();
    }

    MeshBase &NormalizeNormals();

    /// Assigns each vertex in the TriangleMesh the same color \param color.
    MeshBase &PaintUniformColor(const Eigen::Vector3f &color) {
        ResizeAndPaintUniformColor(vertex_colors_, vertices_.size(), color);
        return *this;
    }

protected:
    // Forward child class type to avoid indirect nonvirtual base
    MeshBase(Geometry::GeometryType type);
    MeshBase(Geometry::GeometryType type,
             const utility::device_vector<Eigen::Vector3f> &vertices);
    MeshBase(Geometry::GeometryType type,
             const utility::device_vector<Eigen::Vector3f> &vertices,
             const utility::device_vector<Eigen::Vector3f> &vertex_normals,
             const utility::device_vector<Eigen::Vector3f> &vertex_colors);
    MeshBase(Geometry::GeometryType type,
             const thrust::host_vector<Eigen::Vector3f> &vertices);
public:
    utility::device_vector<Eigen::Vector3f> vertices_;
    utility::device_vector<Eigen::Vector3f> vertex_normals_;
    utility::device_vector<Eigen::Vector3f> vertex_colors_;
};

}
}