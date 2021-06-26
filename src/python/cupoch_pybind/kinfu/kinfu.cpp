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
    // cupoch.kinfu.KinfuOption
    py::class_<kinfu::KinfuOption> kp(m, "KinfuOption", "Parameters for Kinect Fusion.");
    py::detail::bind_default_constructor<kinfu::KinfuOption>(kp);
    py::detail::bind_copy_functions<kinfu::KinfuOption>(kp);
    kp.def_readwrite("num_pyramid_levels", &kinfu::KinfuOption::num_pyramid_levels_)
      .def_readwrite("diameter", &kinfu::KinfuOption::diameter_)
      .def_readwrite("sigma_depth", &kinfu::KinfuOption::sigma_depth_)
      .def_readwrite("sigma_space", &kinfu::KinfuOption::sigma_space_)
      .def_readwrite("depth_cutoff", &kinfu::KinfuOption::depth_cutoff_)
      .def_readwrite("tsdf_length", &kinfu::KinfuOption::tsdf_length_)
      .def_readwrite("tsdf_resolution", &kinfu::KinfuOption::tsdf_resolution_)
      .def_readwrite("sdf_trunc", &kinfu::KinfuOption::sdf_trunc_)
      .def_readwrite("tsdf_color_type", &kinfu::KinfuOption::tsdf_color_type_)
      .def_readwrite("tsdf_origin", &kinfu::KinfuOption::tsdf_origin_)
      .def_readwrite("distance_threshold", &kinfu::KinfuOption::distance_threshold_)
      .def_readwrite("icp_iterations", &kinfu::KinfuOption::icp_iterations_);

    // cupoch.kinfu.KinfuPipeline
    py::class_<kinfu::KinfuPipeline> pipline(
            m, "KinfuPipeline",
            "KinfuPipeline implements Kinect fusion algorithm.");
    py::detail::bind_copy_functions<kinfu::KinfuPipeline>(pipline);
    pipline
        .def(py::init([](const camera::PinholeCameraIntrinsic& intrinsic,
                         const kinfu::KinfuOption& params) {
                return new kinfu::KinfuPipeline(intrinsic, params);
            }))
        .def("reset", &kinfu::KinfuPipeline::Reset)
        .def("process_frame", &kinfu::KinfuPipeline::ProcessFrame)
        .def("extract_point_cloud", &kinfu::KinfuPipeline::ExtractPointCloud)
        .def("extract_triangle_mesh", &kinfu::KinfuPipeline::ExtractTriangleMesh)
        .def_readwrite("cur_pose", &kinfu::KinfuPipeline::cur_pose_)
        .def_readwrite("model_pyramid", &kinfu::KinfuPipeline::model_pyramid_);
}

void pybind_kinfu(py::module &m) {
    py::module m_submodule = m.def_submodule("kinfu");
    pybind_kinfu_classes(m_submodule);
}