#include "cupoch/geometry/laserscanbuffer.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_laserscanbuffer(py::module &m) {
    py::class_<geometry::LaserScanBuffer, PyGeometry3D<geometry::LaserScanBuffer>,
               std::shared_ptr<geometry::LaserScanBuffer>, geometry::GeometryBase<3>>
            laserscan(m, "LaserScanBuffer",
                      "LaserScanBuffer define a sets of scan from a planar laser range-finder.");
    py::detail::bind_copy_functions<geometry::LaserScanBuffer>(laserscan);
    laserscan.def(py::init<int, int, float, float>(),
                  "Create a LaserScanBuffer from given a number of points and angle ranges",
                  "num_steps"_a, "num_max_scans"_a = 10, "min_angle"_a = M_PI, "max_angle"_a = M_PI)
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
             .def("range_filter", &geometry::LaserScanBuffer::RangeFilter)
             .def("scan_shadow_filter", &geometry::LaserScanBuffer::ScanShadowsFilter,
                  "This filter removes laser readings that are most likely caused"
                  " by the veiling effect when the edge of an object is being scanned.",
                  "min_angle"_a, "max_angle"_a, "window"_a,
                  "neighbors"_a = 0, "remove_shadow_start_point"_a = false);
}