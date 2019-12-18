#include "cupoch/io/class_io/trianglemesh_io.h"
#include "cupoch/utility/filesystem.h"
#include "cupoch/utility/console.h"
#include <unordered_map>

using namespace cupoch;
using namespace cupoch::io;

namespace {

static const std::unordered_map<
        std::string,
        std::function<bool(
                const std::string &, geometry::TriangleMesh &, bool)>>
        file_extension_to_trianglemesh_read_function{
                {"ply", ReadTriangleMeshFromPLY},
                {"obj", ReadTriangleMeshFromOBJ},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const geometry::TriangleMesh &,
                           const bool,
                           const bool,
                           const bool,
                           const bool,
                           const bool,
                           const bool)>>
        file_extension_to_trianglemesh_write_function{
                {"ply", WriteTriangleMeshToPLY},
                {"obj", WriteTriangleMeshToOBJ},
        };

}  // unnamed namespace


std::shared_ptr<geometry::TriangleMesh> CreateMeshFromFile(
        const std::string &filename, bool print_progress) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    ReadTriangleMesh(filename, *mesh, print_progress);
    return mesh;
}

bool ReadTriangleMesh(const std::string &filename,
                      geometry::TriangleMesh &mesh,
                      bool print_progress /* = false */) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::TriangleMesh failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_trianglemesh_read_function.find(filename_ext);
    if (map_itr == file_extension_to_trianglemesh_read_function.end()) {
        utility::LogWarning(
                "Read geometry::TriangleMesh failed: unknown file "
                "extension.");
        return false;
    }
    bool success = map_itr->second(filename, mesh, print_progress);
    utility::LogDebug(
            "Read geometry::TriangleMesh: {:d} triangles and {:d} vertices.",
            (int)mesh.triangles_.size(), (int)mesh.vertices_.size());
    if (mesh.HasVertices() && !mesh.HasTriangles()) {
        utility::LogWarning(
                "geometry::TriangleMesh appears to be a geometry::PointCloud "
                "(only contains vertices, but no triangles).");
    }
    return success;
}

bool WriteTriangleMesh(const std::string &filename,
                       const geometry::TriangleMesh &mesh,
                       bool write_ascii /* = false*/,
                       bool compressed /* = false*/,
                       bool write_vertex_normals /* = true*/,
                       bool write_vertex_colors /* = true*/,
                       bool write_triangle_uvs /* = true*/,
                       bool print_progress /* = false*/) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::TriangleMesh failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_trianglemesh_write_function.find(filename_ext);
    if (map_itr == file_extension_to_trianglemesh_write_function.end()) {
        utility::LogWarning(
                "Write geometry::TriangleMesh failed: unknown file "
                "extension.");
        return false;
    }
    bool success = map_itr->second(filename, mesh, write_ascii, compressed,
                                   write_vertex_normals, write_vertex_colors,
                                   write_triangle_uvs, print_progress);
    utility::LogDebug(
            "Write geometry::TriangleMesh: {:d} triangles and {:d} vertices.",
            (int)mesh.triangles_.size(), (int)mesh.vertices_.size());
    return success;
}
