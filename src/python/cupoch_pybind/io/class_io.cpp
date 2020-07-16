/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
**/
#include <string>

#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "cupoch/camera/pinhole_camera_parameters.h"
#include "cupoch/geometry/image.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/io/class_io/ijson_convertible_io.h"
#include "cupoch/io/class_io/image_io.h"
#include "cupoch/io/class_io/pointcloud_io.h"
#include "cupoch/io/class_io/trianglemesh_io.h"
#include "cupoch/io/class_io/voxelgrid_io.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/io/io.h"

using namespace cupoch;

// IO functions have similar arguments, thus the arg docstrings may be shared
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"filename", "Path to file."},
                // Write options
                {"compressed",
                 "Set to ``True`` to write in compressed format."},
                {"format",
                 "The format of the input file. When not specified or set as "
                 "``auto``, the format is inferred from file extension name."},
                {"remove_nan_points",
                 "If true, all points that include a NaN are removed from "
                 "the PointCloud."},
                {"remove_infinite_points",
                 "If true, all points that include an infinite value are "
                 "removed from the PointCloud."},
                {"quality", "Quality of the output file."},
                {"write_ascii",
                 "Set to ``True`` to output in ascii format, otherwise binary "
                 "format will be used."},
                {"write_vertex_normals",
                 "Set to ``False`` to not write any vertex normals, even if "
                 "present on the mesh"},
                {"write_vertex_colors",
                 "Set to ``False`` to not write any vertex colors, even if "
                 "present on the mesh"},
                // Entities
                {"config", "AzureKinectSensor's config file."},
                {"pointcloud", "The ``PointCloud`` object for I/O"},
                {"mesh", "The ``TriangleMesh`` object for I/O"},
                {"line_set", "The ``LineSet`` object for I/O"},
                {"image", "The ``Image`` object for I/O"},
                {"voxel_grid", "The ``VoxelGrid`` object for I/O"},
                {"trajectory",
                 "The ``PinholeCameraTrajectory`` object for I/O"},
                {"intrinsic", "The ``PinholeCameraIntrinsic`` object for I/O"},
                {"parameters",
                 "The ``PinholeCameraParameters`` object for I/O"},
                {"pose_graph", "The ``PoseGraph`` object for I/O"},
                {"feature", "The ``Feature`` object for I/O"},
                {"print_progress",
                 "If set to true a progress bar is visualized in the console"},
};

void pybind_class_io(py::module &m_io) {
    // cupoch::geometry::Image
    m_io.def(
            "read_image",
            [](const std::string &filename) {
                geometry::Image image;
                io::ReadImage(filename, image);
                return image;
            },
            "Function to read Image from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_image",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_image",
            [](const std::string &filename, const geometry::Image &image,
               int quality) {
                return io::WriteImage(filename, image, quality);
            },
            "Function to write Image to file", "filename"_a, "image"_a,
            "quality"_a = 90);
    docstring::FunctionDocInject(m_io, "write_image",
                                 map_shared_argument_docstrings);

    // cupoch::geometry::PointCloud
    m_io.def(
            "read_point_cloud",
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
    docstring::FunctionDocInject(m_io, "read_point_cloud",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_point_cloud",
            [](const std::string &filename,
               const geometry::PointCloud &pointcloud, bool write_ascii,
               bool compressed, bool print_progress) {
                return io::WritePointCloud(filename, pointcloud, write_ascii,
                                           compressed, print_progress);
            },
            "Function to write PointCloud to file", "filename"_a,
            "pointcloud"_a, "write_ascii"_a = false, "compressed"_a = false,
            "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_point_cloud",
                                 map_shared_argument_docstrings);

    // cupoch::geometry::TriangleMesh
    m_io.def(
            "read_triangle_mesh",
            [](const std::string &filename, bool print_progress) {
                geometry::TriangleMesh mesh;
                io::ReadTriangleMesh(filename, mesh, print_progress);
                return mesh;
            },
            "Function to read TriangleMesh from file", "filename"_a,
            "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_triangle_mesh",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_triangle_mesh",
            [](const std::string &filename, const geometry::TriangleMesh &mesh,
               bool write_ascii, bool compressed, bool write_vertex_normals,
               bool write_vertex_colors, bool write_triangle_uvs,
               bool print_progress) {
                return io::WriteTriangleMesh(
                        filename, mesh, write_ascii, compressed,
                        write_vertex_normals, write_vertex_colors,
                        write_triangle_uvs, print_progress);
            },
            "Function to write TriangleMesh to file", "filename"_a, "mesh"_a,
            "write_ascii"_a = false, "compressed"_a = false,
            "write_vertex_normals"_a = true, "write_vertex_colors"_a = true,
            "write_triangle_uvs"_a = true, "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_triangle_mesh",
                                 map_shared_argument_docstrings);

    // cupoch::geometry::VoxelGrid
    m_io.def(
            "read_voxel_grid",
            [](const std::string &filename, const std::string &format,
               bool print_progress) {
                geometry::VoxelGrid voxel_grid;
                io::ReadVoxelGrid(filename, voxel_grid, format);
                return voxel_grid;
            },
            "Function to read VoxelGrid from file", "filename"_a,
            "format"_a = "auto", "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "read_voxel_grid",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_voxel_grid",
            [](const std::string &filename,
               const geometry::VoxelGrid &voxel_grid, bool write_ascii,
               bool compressed, bool print_progress) {
                return io::WriteVoxelGrid(filename, voxel_grid, write_ascii,
                                          compressed, print_progress);
            },
            "Function to write VoxelGrid to file", "filename"_a, "voxel_grid"_a,
            "write_ascii"_a = false, "compressed"_a = false,
            "print_progress"_a = false);
    docstring::FunctionDocInject(m_io, "write_voxel_grid",
                                 map_shared_argument_docstrings);

    // cupoch::camera
    m_io.def(
            "read_pinhole_camera_intrinsic",
            [](const std::string &filename) {
                camera::PinholeCameraIntrinsic intrinsic;
                io::ReadIJsonConvertible(filename, intrinsic);
                return intrinsic;
            },
            "Function to read PinholeCameraIntrinsic from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_pinhole_camera_intrinsic",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_pinhole_camera_intrinsic",
            [](const std::string &filename,
               const camera::PinholeCameraIntrinsic &intrinsic) {
                return io::WriteIJsonConvertible(filename, intrinsic);
            },
            "Function to write PinholeCameraIntrinsic to file", "filename"_a,
            "intrinsic"_a);
    docstring::FunctionDocInject(m_io, "write_pinhole_camera_intrinsic",
                                 map_shared_argument_docstrings);

    m_io.def(
            "read_pinhole_camera_parameters",
            [](const std::string &filename) {
                camera::PinholeCameraParameters parameters;
                io::ReadIJsonConvertible(filename, parameters);
                return parameters;
            },
            "Function to read PinholeCameraParameters from file", "filename"_a);
    docstring::FunctionDocInject(m_io, "read_pinhole_camera_parameters",
                                 map_shared_argument_docstrings);

    m_io.def(
            "write_pinhole_camera_parameters",
            [](const std::string &filename,
               const camera::PinholeCameraParameters &parameters) {
                return io::WriteIJsonConvertible(filename, parameters);
            },
            "Function to write PinholeCameraParameters to file", "filename"_a,
            "parameters"_a);
    docstring::FunctionDocInject(m_io, "write_pinhole_camera_parameters",
                                 map_shared_argument_docstrings);
}