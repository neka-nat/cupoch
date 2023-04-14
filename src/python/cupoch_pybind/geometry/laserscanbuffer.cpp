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
#include "cupoch/geometry/laserscanbuffer.h"

#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_laserscanbuffer(py::module &m) {
    py::class_<geometry::LaserScanBuffer, PyGeometry3D<geometry::LaserScanBuffer>,
               std::shared_ptr<geometry::LaserScanBuffer>, geometry::GeometryBase3D>
            laserscan(m, "LaserScanBuffer",
                      "LaserScanBuffer define a sets of scan from a planar laser range-finder.");
    py::detail::bind_copy_functions<geometry::LaserScanBuffer>(laserscan);
    laserscan.def(py::init<int, int, float, float>(),
                  "Create a LaserScanBuffer from given a number of points and angle ranges",
                  "num_steps"_a, "num_max_scans"_a = 10, "min_angle"_a = M_PI, "max_angle"_a = M_PI)
             .def("is_full", &geometry::LaserScanBuffer::IsFull)
             .def("add_ranges", [](geometry::LaserScanBuffer &self,
                                   const wrapper::device_vector_float &ranges,
                                   const Eigen::Matrix4f &transformation,
                                   const wrapper::device_vector_float &intensities) {
                                       return self.AddRanges(ranges.data_, transformation, intensities.data_);
                                   },
                  "Add single scan ranges",
                  py::arg("ranges"),
                  py::arg("transformation") = Eigen::Matrix4f::Identity(),
                  py::arg("intensities") = wrapper::device_vector_float())
             .def("add_host_ranges", [](geometry::LaserScanBuffer &self,
                                        const cupoch::utility::pinned_host_vector<float> &ranges,
                                        const Eigen::Matrix4f &transformation,
                                        const cupoch::utility::pinned_host_vector<float> &intensities) {
                                            return self.AddRanges(
                                                wrapper::device_vector_float(ranges).data_,
                                                transformation,
                                                wrapper::device_vector_float(intensities).data_
                                            );
                                        },
                  "Add host single scan ranges",
                  py::arg("ranges"),
                  py::arg("transformation") = Eigen::Matrix4f::Identity(),
                  py::arg("intensities") = cupoch::utility::pinned_host_vector<float>())
             .def("merge", &geometry::LaserScanBuffer::Merge,
                  "Merge other LaserScanBuffer into this one")
             .def("pop_one_scan", &geometry::LaserScanBuffer::PopOneScan,
                  "Pop one scan from the buffer")
             .def("pop_host_one_scan", &geometry::LaserScanBuffer::PopHostOneScan,
                  "Pop host one scan from the buffer")
             .def("range_filter", &geometry::LaserScanBuffer::RangeFilter)
             .def("scan_shadow_filter", &geometry::LaserScanBuffer::ScanShadowsFilter,
                  "This filter removes laser readings that are most likely caused"
                  " by the veiling effect when the edge of an object is being scanned.",
                  "min_angle"_a, "max_angle"_a, "window"_a,
                  "neighbors"_a = 0, "remove_shadow_start_point"_a = false)
            .def_static("create_from_point_cloud", &geometry::LaserScanBuffer::CreateFromPointCloud,
                        "Create a LaserScanBuffer from a point cloud",
                        "pcd"_a,
                        "angle_increment"_a,
                        "min_height"_a,
                        "max_height"_a,
                        "num_vertical_divisions"_a = 1,
                        "min_range"_a = 0.0,
                        "max_range"_a = std::numeric_limits<float>::infinity(),
                        "min_angle"_a = -M_PI,
                        "max_angle"_a = M_PI)
            .def_static("create_from_depth_image", &geometry::LaserScanBuffer::CreateFromDepthImage,
                        "Create a LaserScanBuffer from a depth image",
                        "depth"_a,
                        "intrinsic"_a,
                        "angle_increment"_a,
                        "min_y"_a,
                        "max_y"_a,
                        "num_vertical_divisions"_a = 1,
                        "min_range"_a = 0.0,
                        "max_range"_a = std::numeric_limits<float>::infinity(),
                        "min_angle"_a = -M_PI,
                        "max_angle"_a = M_PI,
                        "depth_scale"_a = 1000.0,
                        "depth_trunc"_a = 1000.0,
                        "stride"_a = 1)
            .def_readonly("num_steps", &geometry::LaserScanBuffer::num_steps_,
                          "Integer: Number of steps per scan.")
            .def_readonly("num_max_scans", &geometry::LaserScanBuffer::num_max_scans_,
                          "Integer: Maximum buffer size.")
            .def_readwrite("min_angle", &geometry::LaserScanBuffer::min_angle_)
            .def_readwrite("max_angle", &geometry::LaserScanBuffer::max_angle_)
            .def_property_readonly(
                    "num_scans", &geometry::LaserScanBuffer::GetNumScans,
                    "Integer: Number of scans in the buffer.")
            .def_property_readonly(
                    "ranges",
                    [](geometry::LaserScanBuffer &scan) {
                        return wrapper::device_vector_float(scan.ranges_);
                    })
            .def_property_readonly(
                    "intensities",
                    [](geometry::LaserScanBuffer &scan) {
                        return wrapper::device_vector_float(scan.intensities_);
                    });
}