#pragma once

#include <Eigen/Core>
#include <string>
#include <thrust/host_vector.h>
#include "cupoch/io/class_io/image_io.h"

namespace cupoch {

namespace geometry {
class TriangleMesh;
}

namespace io {

struct HostTriangleMesh {
    HostTriangleMesh() = default;
    ~HostTriangleMesh() = default;
    void FromDevice(const geometry::TriangleMesh& mesh);
    void ToDevice(geometry::TriangleMesh& mesh) const;
    void Clear();
    thrust::host_vector<Eigen::Vector3f> vertices_;
    thrust::host_vector<Eigen::Vector3f> vertex_normals_;
    thrust::host_vector<Eigen::Vector3f> vertex_colors_;
    thrust::host_vector<Eigen::Vector3i> triangles_;
    thrust::host_vector<Eigen::Vector3f> triangle_normals_;
    thrust::host_vector<Eigen::Vector2f> triangle_uvs_;
    HostImage texture_;
};

/// Factory function to create a mesh from a file (TriangleMeshFactory.cpp)
/// Return an empty mesh if fail to read the file.
std::shared_ptr<geometry::TriangleMesh> CreateMeshFromFile(
        const std::string &filename, bool print_progress = false);

/// The general entrance for reading a TriangleMesh from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ReadTriangleMesh(const std::string &filename,
                      geometry::TriangleMesh &mesh,
                      bool print_progress = false);

/// The general entrance for writing a TriangleMesh to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports binary encoding and compression, the later
/// two parameters will be used. Otherwise they will be ignored.
/// At current only .obj format supports uv coordinates (triangle_uvs) and
/// textures.
/// \return return true if the write function is successful, false otherwise.
bool WriteTriangleMesh(const std::string &filename,
                       const geometry::TriangleMesh &mesh,
                       bool write_ascii = false,
                       bool compressed = false,
                       bool write_vertex_normals = true,
                       bool write_vertex_colors = true,
                       bool write_triangle_uvs = true,
                       bool print_progress = false);

bool ReadTriangleMeshFromPLY(const std::string &filename,
                             geometry::TriangleMesh &mesh,
                             bool print_progress);

bool WriteTriangleMeshToPLY(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            bool write_ascii,
                            bool compressed,
                            bool write_vertex_normals,
                            bool write_vertex_colors,
                            bool write_triangle_uvs,
                            bool print_progress);


bool ReadTriangleMeshFromOBJ(const std::string &filename,
                             geometry::TriangleMesh &mesh,
                             bool print_progress);

bool WriteTriangleMeshToOBJ(const std::string &filename,
                            const geometry::TriangleMesh &mesh,
                            bool write_ascii,
                            bool compressed,
                            bool write_vertex_normals,
                            bool write_vertex_colors,
                            bool write_triangle_uvs,
                            bool print_progress);

/// Function to convert a polygon into a collection of
/// triangles whose vertices are only those of the polygon.
/// Assume that the vertices are connected by edges based on their order, and
/// the final vertex connected to the first.
/// The triangles are added to the mesh that is passed as reference. The mesh
/// should contain all vertices prior to calling this function.
/// \return return true if triangulation is successful, false otherwise.
bool AddTrianglesByEarClipping(HostTriangleMesh &mesh,
                               std::vector<unsigned int> &indices);

}  // namespace io
}  // namespace cupoch