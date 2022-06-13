/**
 * Copyright (c) 2021 Neka-Nat
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
#include "cupoch/geometry/keypoint.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch_pybind/cupoch_pybind.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"

using namespace cupoch;

void pybind_keypoint_methods(py::module &m) {
    m.def("compute_iss_keypoints", [] (
              const geometry::PointCloud& input,
              float salient_radius,
              float non_max_radius,
              float gamma_21,
              float gamma_32,
              int min_neighbors,
              int max_neighbors) {
              auto res = geometry::keypoint::ComputeISSKeypoints(input, salient_radius, non_max_radius, gamma_21, gamma_32, min_neighbors, max_neighbors);
              return std::make_tuple(std::get<0>(res), wrapper::device_vector_bool(*std::get<1>(res)));
          },
          "Function that computes the ISS keypoints from an input point "
          "cloud. This implements the keypoint detection modules "
          "proposed in Yu Zhong, 'Intrinsic Shape Signatures: A Shape "
          "Descriptor for 3D Object Recognition', 2009.",
          "input"_a, "salient_radius"_a = 0.0, "non_max_radius"_a = 0.0,
          "gamma_21"_a = 0.975, "gamma_32"_a = 0.975, "min_neighbors"_a = 5,
          "max_neighbots"_a = geometry::NUM_MAX_NN);

    docstring::FunctionDocInject(
            m, "compute_iss_keypoints",
            {{"input", "The Input point cloud."},
             {"salient_radius",
              "The radius of the spherical neighborhood used to detect "
              "keypoints."},
             {"non_max_radius", "The non maxima supression radius"},
             {"gamma_21",
              "The upper bound on the ratio between the second and the "
              "first "
              "eigenvalue returned by the EVD"},
             {"gamma_32",
              "The upper bound on the ratio between the third and the "
              "second "
              "eigenvalue returned by the EVD"},
             {"min_neighbors",
              "Minimum number of neighbors that has to be found to "
              "consider a "
              "keypoint"},
             {"max_neighbors",
              "Maximum number of neighbors that has to be found to "
              "consider a "
              "keypoint"}});
}

void pybind_keypoint(py::module &m) {
    py::module m_submodule = m.def_submodule("keypoint", "Keypoint Detectors.");
    pybind_keypoint_methods(m_submodule);
}
