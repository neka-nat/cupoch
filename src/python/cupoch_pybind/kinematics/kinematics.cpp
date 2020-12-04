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
#include "cupoch_pybind/kinematics/kinematics.h"

#include "cupoch/kinematics/kinematic_chain.h"

using namespace cupoch;

void pybind_kinematics_classes(py::module &m) {
    // cupoch.registration.ICPConvergenceCriteria
    py::class_<kinematics::KinematicChain> kinematic_chain(
            m, "KinematicChain",
            "KinematicChain implements kinematic 3d model from URDF file.");
    py::detail::bind_copy_functions<kinematics::KinematicChain>(kinematic_chain);
    kinematic_chain
        .def(py::init([](const std::string& filename) {
                return new kinematics::KinematicChain(filename);
            }))
        .def("forward_kinematics", &kinematics::KinematicChain::ForwardKinematics,
             "Calculate forward kinematics and get the link poses",
             "jmap"_a = kinematics::KinematicChain::JointMap(),
             "base"_a = Eigen::Matrix4f::Identity())
        .def("get_transformed_visual_geometries", &kinematics::KinematicChain::GetTransformedVisualGeometries);
}

void pybind_kinematics(py::module &m) {
    py::module m_submodule = m.def_submodule("kinematics");
    pybind_kinematics_classes(m_submodule);
}