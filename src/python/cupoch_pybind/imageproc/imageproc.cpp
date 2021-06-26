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
#include "cupoch_pybind/imageproc/imageproc.h"

#include "cupoch/imageproc/sgm.h"
#include "cupoch_pybind/docstring.h"

using namespace cupoch;

void pybind_imageproc_classes(py::module &m) {
    py::enum_<imageproc::SGMOption::DisparitySizeType> disp_size_t(m, "DisparitySizeType");
    disp_size_t.value("DisparitySize64", imageproc::SGMOption::DisparitySizeType::DisparitySize64)
               .value("DisparitySize128", imageproc::SGMOption::DisparitySizeType::DisparitySize128)
               .value("DisparitySize256", imageproc::SGMOption::DisparitySizeType::DisparitySize256)
               .export_values();
    py::enum_<imageproc::SGMOption::PathType> path_t(m, "PathType");
    path_t.value("ScanPath4", imageproc::SGMOption::PathType::ScanPath4)
          .value("ScanPath8", imageproc::SGMOption::PathType::ScanPath8)
          .export_values();
    // cupoch.imageproc.SGMOption
    py::class_<imageproc::SGMOption> so(m, "SGMOption", "Parameters for Semi-Global Matching.");
    py::detail::bind_default_constructor<imageproc::SGMOption>(so);
    py::detail::bind_copy_functions<imageproc::SGMOption>(so);
    so.def_readwrite("width", &imageproc::SGMOption::width_)
      .def_readwrite("height", &imageproc::SGMOption::height_)
      .def_readwrite("p1", &imageproc::SGMOption::p1_)
      .def_readwrite("p2", &imageproc::SGMOption::p2_)
      .def_readwrite("uniqueness", &imageproc::SGMOption::uniqueness_)
      .def_readwrite("disp_size", &imageproc::SGMOption::disp_size_)
      .def_readwrite("path_type", &imageproc::SGMOption::path_type_)
      .def_readwrite("min_disp", &imageproc::SGMOption::min_disp_)
      .def_readwrite("lr_max_diff", &imageproc::SGMOption::lr_max_diff_);

    // cupoch.imageproc.SemiGlobalMatching
    py::class_<imageproc::SemiGlobalMatching> sgm(
            m, "SemiGlobalMatching",
            "This class implements Semi-Global Matching algorithm.");
    sgm.def(py::init([](const imageproc::SGMOption& params) {
                return new imageproc::SemiGlobalMatching(params);
            }))
       .def("process_frame", &imageproc::SemiGlobalMatching::ProcessFrame);
}

void pybind_imageproc(py::module &m) {
    py::module m_submodule = m.def_submodule("imageproc");
    pybind_imageproc_classes(m_submodule);
}