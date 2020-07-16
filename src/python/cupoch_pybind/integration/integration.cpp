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
#include "cupoch_pybind/integration/integration.h"

#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/integration/tsdfvolume.h"
#include "cupoch/integration/uniform_tsdfvolume.h"
#include "cupoch_pybind/docstring.h"

using namespace cupoch;

template <class TSDFVolumeBase = integration::TSDFVolume>
class PyTSDFVolume : public TSDFVolumeBase {
public:
    using TSDFVolumeBase::TSDFVolumeBase;
    void Reset() override { PYBIND11_OVERLOAD_PURE(void, TSDFVolumeBase, ); }
    void Integrate(const geometry::RGBDImage &image,
                   const camera::PinholeCameraIntrinsic &intrinsic,
                   const Eigen::Matrix4f &extrinsic) override {
        PYBIND11_OVERLOAD_PURE(void, TSDFVolumeBase, image, intrinsic,
                               extrinsic);
    }
    std::shared_ptr<geometry::PointCloud> ExtractPointCloud() override {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<geometry::PointCloud>,
                               TSDFVolumeBase, );
    }
    std::shared_ptr<geometry::TriangleMesh> ExtractTriangleMesh() override {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<geometry::TriangleMesh>,
                               TSDFVolumeBase, );
    }
};

void pybind_integration_classes(py::module &m) {
    // cupoch.integration.TSDFVolumeColorType
    py::enum_<integration::TSDFVolumeColorType> tsdf_volume_color_type(
            m, "TSDFVolumeColorType", py::arithmetic());
    tsdf_volume_color_type
            .value("NoColor", integration::TSDFVolumeColorType::NoColor)
            .value("RGB8", integration::TSDFVolumeColorType::RGB8)
            .value("Gray32", integration::TSDFVolumeColorType::Gray32)
            .export_values();
    // Trick to write docs without listing the members in the enum class again.
    tsdf_volume_color_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for TSDFVolumeColorType.";
            }),
            py::none(), py::none(), "");

    // cupoch.integration.TSDFVolume
    py::class_<integration::TSDFVolume, PyTSDFVolume<integration::TSDFVolume>>
            tsdfvolume(m, "TSDFVolume", R"(Base class of the Truncated
Signed Distance Function (TSDF) volume This volume is usually used to integrate
surface data (e.g., a series of RGB-D images) into a Mesh or PointCloud. The
basic technique is presented in the following paper:
A volumetric method for building complex models from range images
B. Curless and M. Levoy
In SIGGRAPH, 1996)");
    tsdfvolume
            .def("reset", &integration::TSDFVolume::Reset,
                 "Function to reset the integration::TSDFVolume")
            .def("integrate", &integration::TSDFVolume::Integrate,
                 "Function to integrate an RGB-D image into the volume",
                 "image"_a, "intrinsic"_a, "extrinsic"_a)
            .def("extract_point_cloud",
                 &integration::TSDFVolume::ExtractPointCloud,
                 "Function to extract a point cloud with normals")
            .def("extract_triangle_mesh",
                 &integration::TSDFVolume::ExtractTriangleMesh,
                 "Function to extract a triangle mesh")
            .def_readwrite("voxel_length",
                           &integration::TSDFVolume::voxel_length_,
                           "float: Length of the voxel in meters.")
            .def_readwrite("sdf_trunc", &integration::TSDFVolume::sdf_trunc_,
                           "float: Truncation value for signed distance "
                           "function (SDF).")
            .def_readwrite("color_type", &integration::TSDFVolume::color_type_,
                           "integration.TSDFVolumeColorType: Color type of the "
                           "TSDF volume.");
    docstring::ClassMethodDocInject(m, "TSDFVolume", "extract_point_cloud");
    docstring::ClassMethodDocInject(m, "TSDFVolume", "extract_triangle_mesh");
    docstring::ClassMethodDocInject(
            m, "TSDFVolume", "integrate",
            {{"image", "RGBD image."},
             {"intrinsic", "Pinhole camera intrinsic parameters."},
             {"extrinsic", "Extrinsic parameters."}});
    docstring::ClassMethodDocInject(m, "TSDFVolume", "reset");

    // cupoch.integration.UniformTSDFVolume: cupoch.integration.TSDFVolume
    py::class_<integration::UniformTSDFVolume,
               PyTSDFVolume<integration::UniformTSDFVolume>,
               integration::TSDFVolume>
            uniform_tsdfvolume(
                    m, "UniformTSDFVolume",
                    "UniformTSDFVolume implements the classic TSDF "
                    "volume with uniform voxel grid (Curless and Levoy 1996).");
    py::detail::bind_copy_functions<integration::UniformTSDFVolume>(
            uniform_tsdfvolume);
    uniform_tsdfvolume
            .def(py::init([](float length, int resolution, float sdf_trunc,
                             integration::TSDFVolumeColorType color_type) {
                     return new integration::UniformTSDFVolume(
                             length, resolution, sdf_trunc, color_type);
                 }),
                 "length"_a, "resolution"_a, "sdf_trunc"_a, "color_type"_a)
            .def("__repr__",
                 [](const integration::UniformTSDFVolume &vol) {
                     return std::string("integration::UniformTSDFVolume ") +
                            (vol.color_type_ ==
                                             integration::TSDFVolumeColorType::
                                                     NoColor
                                     ? std::string("without color.")
                                     : std::string("with color."));
                 })  // todo: extend
            .def("extract_voxel_point_cloud",
                 &integration::UniformTSDFVolume::ExtractVoxelPointCloud,
                 "Debug function to extract the voxel data into a point cloud.")
            .def("extract_voxel_grid",
                 &integration::UniformTSDFVolume::ExtractVoxelGrid,
                 "Debug function to extract the voxel data VoxelGrid.")
            .def_readwrite("length", &integration::UniformTSDFVolume::length_,
                           "Total length, where ``voxel_length = length / "
                           "resolution``.")
            .def_readwrite("resolution",
                           &integration::UniformTSDFVolume::resolution_,
                           "Resolution over the total length, where "
                           "``voxel_length = length / resolution``");
    docstring::ClassMethodDocInject(m, "UniformTSDFVolume",
                                    "extract_voxel_point_cloud");
}

void pybind_integration_methods(py::module &m) {
    // Currently empty
}

void pybind_integration(py::module &m) {
    py::module m_submodule = m.def_submodule("integration");
    pybind_integration_classes(m_submodule);
    pybind_integration_methods(m_submodule);
}