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
#include "cupoch_pybind/kinfu/kinfu.h"

#include "cupoch/kinfu/kinfu.h"

using namespace cupoch;

void pybind_kinfu_classes(py::module &m) {
    // cupoch.kinfu.KinfuParameters
    py::class_<kinfu::KinfuParameters> kp(m, "KinfuParameters", "Parameters for Kinect Fusion.");
    py::detail::bind_default_constructor<kinfu::KinfuParameters>(kp);
    py::detail::bind_copy_functions<kinfu::KinfuParameters>(kp);
    kp.def_readwrite("num_pyramid_levels", &kinfu::KinfuParameters::num_pyramid_levels_)
      .def_readwrite("diameter", &kinfu::KinfuParameters::diameter_)
      .def_readwrite("sigma_depth", &kinfu::KinfuParameters::sigma_depth_)
      .def_readwrite("sigma_space", &kinfu::KinfuParameters::sigma_space_)
      .def_readwrite("depth_cutoff", &kinfu::KinfuParameters::depth_cutoff_)
      .def_readwrite("tsdf_length", &kinfu::KinfuParameters::tsdf_length_)
      .def_readwrite("tsdf_resolution", &kinfu::KinfuParameters::tsdf_resolution_)
      .def_readwrite("sdf_trunc", &kinfu::KinfuParameters::sdf_trunc_)
      .def_readwrite("tsdf_color_type", &kinfu::KinfuParameters::tsdf_color_type_)
      .def_readwrite("tsdf_origin", &kinfu::KinfuParameters::tsdf_origin_)
      .def_readwrite("distance_threshold", &kinfu::KinfuParameters::distance_threshold_)
      .def_readwrite("icp_iterations", &kinfu::KinfuParameters::icp_iterations_);

    // cupoch.kinfu.Pipeline
    py::class_<kinfu::Pipeline> pipline(
            m, "Pipeline",
            "Pipeline implements Kinect fusion algorithm.");
    py::detail::bind_copy_functions<kinfu::Pipeline>(pipline);
    pipline
        .def(py::init([](const camera::PinholeCameraIntrinsic& intrinsic,
                         const kinfu::KinfuParameters& params) {
                return new kinfu::Pipeline(intrinsic, params);
            }))
        .def("reset", &kinfu::Pipeline::Reset)
        .def("process_frame", &kinfu::Pipeline::ProcessFrame)
        .def("extract_point_cloud", &kinfu::Pipeline::ExtractPointCloud)
        .def("extract_triangle_mesh", &kinfu::Pipeline::ExtractTriangleMesh)
        .def_readwrite("cur_pose", &kinfu::Pipeline::cur_pose_)
        .def_readwrite("model_pyramid", &kinfu::Pipeline::model_pyramid_);
}

void pybind_kinfu(py::module &m) {
    py::module m_submodule = m.def_submodule("kinfu");
    pybind_kinfu_classes(m_submodule);
}