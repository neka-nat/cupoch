#pragma once
#include "cupoch/geometry/mesh_base.h"
#include "cupoch/geometry/image.h"


namespace cupoch {
namespace geometry {

class TriangleMesh : public MeshBase {
public:
    TriangleMesh();
    TriangleMesh(const thrust::device_vector<Eigen::Vector3f> &vertices,
                 const thrust::device_vector<Eigen::Vector3i> &triangles);
    ~TriangleMesh() override;

    thrust::host_vector<Eigen::Vector3i> GetTriangles() const;
    void SetTriangles(const thrust::host_vector<Eigen::Vector3i>& triangles);

    thrust::host_vector<Eigen::Vector3f> GetTriangleNormals() const;
    void SetTriangleNormals(const thrust::host_vector<Eigen::Vector3f>& triangle_normals);

    thrust::host_vector<int> GetAdjacencyMatrix() const;
    void SetAdjacencyMatrix(const thrust::host_vector<int>& adjacency_matrix);

    thrust::host_vector<Eigen::Vector2f> GetTriangleUVs() const;
    void SetTriangleUVs(thrust::host_vector<Eigen::Vector2f>& triangle_uvs);

public:
    virtual TriangleMesh &Clear() override;
    TriangleMesh &operator+=(const TriangleMesh &mesh);
    TriangleMesh operator+(const TriangleMesh &mesh) const;

    __host__ __device__
    bool HasTriangles() const {
        return vertices_.size() > 0 && triangles_.size() > 0;
    }

    __host__ __device__
    bool HasTriangleNormals() const {
        return HasTriangles() && triangles_.size() == triangle_normals_.size();
    }

    __host__ __device__
    bool HasAdjacencyMatrix() const {
        return vertices_.size() > 0 &&
               adjacency_matrix_.size() == vertices_.size() * vertices_.size();
    }

    __host__ __device__
    bool HasTriangleUvs() const {
        return HasTriangles() && triangle_uvs_.size() == 3 * triangles_.size();
    }

    bool HasTexture() const { return !texture_.IsEmpty(); }

    TriangleMesh &NormalizeNormals();

    /// Function to compute triangle normals, usually called before rendering
    TriangleMesh &ComputeTriangleNormals(bool normalized = true);

    /// Function to compute vertex normals, usually called before rendering
    TriangleMesh &ComputeVertexNormals(bool normalized = true);

    /// Function to compute adjacency matrix, call before adjacency matrix is needed
    TriangleMesh &ComputeAdjacencyMatrix();

    /// Factory function to create a sphere mesh (TriangleMeshFactory.cpp)
    /// The sphere with \param radius will be centered at (0, 0, 0).
    /// Its axis is aligned with z-axis.
    /// The longitudes will be split into \param resolution segments.
    /// The latitudes will be split into \param resolution * 2 segments.
    static std::shared_ptr<TriangleMesh> CreateSphere(float radius = 1.0,
                                                      int resolution = 20);


    /// Factory function to create a cylinder mesh (TriangleMeshFactory.cpp)
    /// The axis of the cylinder will be from (0, 0, -height/2) to (0, 0,
    /// height/2). The circle with \param radius will be split into \param
    /// resolution segments. The \param height will be split into \param split
    /// segments.
    static std::shared_ptr<TriangleMesh> CreateCylinder(float radius = 1.0,
                                                        float height = 2.0,
                                                        int resolution = 20,
                                                        int split = 4);

    /// Factory function to create a cone mesh (TriangleMeshFactory.cpp)
    /// The axis of the cone will be from (0, 0, 0) to (0, 0, \param height).
    /// The circle with \param radius will be split into \param resolution
    /// segments. The height will be split into \param split segments.
    static std::shared_ptr<TriangleMesh> CreateCone(float radius = 1.0,
                                                    float height = 2.0,
                                                    int resolution = 20,
                                                    int split = 1);

    /// Factory function to create an arrow mesh (TriangleMeshFactory.cpp)
    /// The axis of the cone with \param cone_radius will be along the z-axis.
    /// The cylinder with \param cylinder_radius is from
    /// (0, 0, 0) to (0, 0, cylinder_height), and
    /// the cone is from (0, 0, cylinder_height)
    /// to (0, 0, cylinder_height + cone_height).
    /// The cone will be split into \param resolution segments.
    /// The \param cylinder_height will be split into \param cylinder_split
    /// segments. The \param cone_height will be split into \param cone_split
    /// segments.
    static std::shared_ptr<TriangleMesh> CreateArrow(
            float cylinder_radius = 1.0,
            float cone_radius = 1.5,
            float cylinder_height = 5.0,
            float cone_height = 4.0,
            int resolution = 20,
            int cylinder_split = 4,
            int cone_split = 1);

    /// Factory function to create a coordinate frame mesh
    /// (TriangleMeshFactory.cu) The coordinate frame will be centered at
    /// \param origin The x, y, z axis will be rendered as red, green, and blue
    /// arrows respectively. \param size is the length of the axes.
    static std::shared_ptr<TriangleMesh> CreateCoordinateFrame(
            float size = 1.0,
            const Eigen::Vector3f &origin = Eigen::Vector3f(0.0, 0.0, 0.0));

public:
    thrust::device_vector<Eigen::Vector3i> triangles_;
    thrust::device_vector<Eigen::Vector3f> triangle_normals_;
    thrust::device_vector<int> adjacency_matrix_;
    thrust::device_vector<Eigen::Vector2f> triangle_uvs_;
    Image texture_;
};

}
}