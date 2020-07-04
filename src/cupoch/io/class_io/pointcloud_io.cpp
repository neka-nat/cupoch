#include "cupoch/io/class_io/pointcloud_io.h"

#include <unordered_map>

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/filesystem.h"
#include "cupoch/utility/platform.h"

namespace cupoch {

namespace {
using namespace io;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::PointCloud &, bool)>>
        file_extension_to_pointcloud_read_function{
                {"ply", ReadPointCloudFromPLY},
                {"pcd", ReadPointCloudFromPCD},
        };

static const std::unordered_map<std::string,
                                std::function<bool(const std::string &,
                                                   const geometry::PointCloud &,
                                                   const bool,
                                                   const bool,
                                                   const bool)>>
        file_extension_to_pointcloud_write_function{
                {"ply", WritePointCloudToPLY},
                {"pcd", WritePointCloudToPCD},
        };
}  // unnamed namespace

namespace io {

std::shared_ptr<geometry::PointCloud> CreatePointCloudFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    ReadPointCloud(filename, *pointcloud, format, print_progress);
    return pointcloud;
}

bool ReadPointCloud(const std::string &filename,
                    geometry::PointCloud &pointcloud,
                    const std::string &format,
                    bool remove_nan_points,
                    bool remove_infinite_points,
                    bool print_progress) {
    std::string filename_ext;
    if (format == "auto") {
        filename_ext =
                utility::filesystem::GetFileExtensionInLowerCase(filename);
    } else {
        filename_ext = format;
    }
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::PointCloud failed: unknown file extension.\n");
        return false;
    }
    auto map_itr =
            file_extension_to_pointcloud_read_function.find(filename_ext);
    if (map_itr == file_extension_to_pointcloud_read_function.end()) {
        utility::LogWarning(
                "Read geometry::PointCloud failed: unknown file extension.\n");
        return false;
    }
    bool success = map_itr->second(filename, pointcloud, print_progress);
    utility::LogDebug("Read geometry::PointCloud: {:d} vertices.\n",
                      (int)pointcloud.GetPoints().size());
    if (remove_nan_points || remove_infinite_points) {
        pointcloud.RemoveNoneFinitePoints(remove_nan_points,
                                          remove_infinite_points);
    }
    return success;
}

bool WritePointCloud(const std::string &filename,
                     const geometry::PointCloud &pointcloud,
                     bool write_ascii /* = false*/,
                     bool compressed /* = false*/,
                     bool print_progress) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::PointCloud failed: unknown file extension.\n");
        return false;
    }
    auto map_itr =
            file_extension_to_pointcloud_write_function.find(filename_ext);
    if (map_itr == file_extension_to_pointcloud_write_function.end()) {
        utility::LogWarning(
                "Write geometry::PointCloud failed: unknown file extension.\n");
        return false;
    }
    bool success = map_itr->second(filename, pointcloud, write_ascii,
                                   compressed, print_progress);
    utility::LogDebug("Write geometry::PointCloud: {:d} vertices.\n",
                      (int)pointcloud.points_.size());
    return success;
}

}  // namespace io
}  // namespace cupoch