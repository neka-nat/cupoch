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

    thrust::host_vector<int> GetAdjacencyList() const;
    void SetAdjacencyList(const thrust::host_vector<int>& adjacency_list);

    thrust::host_vector<Eigen::Vector2f> GetTriangleUVs() const;
    void SetTriangleUVs(thrust::host_vector<Eigen::Vector2f>& triangle_uvs);

public:
    virtual TriangleMesh &Clear() override;

    __host__ __device__
    bool HasTriangles() const {
        return vertices_.size() > 0 && triangles_.size() > 0;
    }

    __host__ __device__
    bool HasTriangleNormals() const {
        return HasTriangles() && triangles_.size() == triangle_normals_.size();
    }

    __host__ __device__
    bool HasAdjacencyList() const {
        return vertices_.size() > 0 &&
               adjacency_list_.size() == vertices_.size();
    }

    __host__ __device__
    bool HasTriangleUvs() const {
        return HasTriangles() && triangle_uvs_.size() == 3 * triangles_.size();
    }

    bool HasTexture() const { return !texture_.IsEmpty(); }

public:
    thrust::device_vector<Eigen::Vector3i> triangles_;
    thrust::device_vector<Eigen::Vector3f> triangle_normals_;
    thrust::device_vector<int> adjacency_list_;
    thrust::device_vector<Eigen::Vector2f> triangle_uvs_;
    Image texture_;
};

}
}