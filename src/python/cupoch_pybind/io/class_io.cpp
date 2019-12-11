#include "cupoch_pybind/io/io.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/io/class_io/pointcloud_io.h"

#include <string>

using namespace cupoch;

void pybind_class_io(py::module &m_io) {
    // cupoch::geometry::PointCloud
    m_io.def("read_point_cloud",
             [](const std::string &filename, const std::string &format,
                bool remove_nan_points, bool remove_infinite_points,
                bool print_progress) {
                 geometry::PointCloud pcd;
                 io::ReadPointCloud(filename, pcd, format, remove_nan_points,
                                    remove_infinite_points, print_progress);
                 return pcd;
             },
             "Function to read PointCloud from file", "filename"_a,
             "format"_a = "auto", "remove_nan_points"_a = true,
             "remove_infinite_points"_a = true, "print_progress"_a = false);

    m_io.def("write_point_cloud",
             [](const std::string &filename,
                const geometry::PointCloud &pointcloud, bool write_ascii,
                bool compressed, bool print_progress) {
                 return io::WritePointCloud(filename, pointcloud, write_ascii,
                                            compressed, print_progress);
             },
             "Function to write PointCloud to file", "filename"_a,
             "pointcloud"_a, "write_ascii"_a = false, "compressed"_a = false,
             "print_progress"_a = false);
}