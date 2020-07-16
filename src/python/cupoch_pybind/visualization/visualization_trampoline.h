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
#pragma once

#include "cupoch/visualization/visualizer/view_control.h"
#include "cupoch/visualization/visualizer/visualizer.h"
#include "cupoch_pybind/cupoch_pybind.h"

using namespace cupoch;

template <class VisualizerBase = visualization::Visualizer>
class PyVisualizer : public VisualizerBase {
public:
    using VisualizerBase::VisualizerBase;
    bool AddGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr,
                     bool reset_bounding_box = true) override {
        PYBIND11_OVERLOAD(bool, VisualizerBase, AddGeometry, geometry_ptr);
    }
    bool UpdateGeometry(std::shared_ptr<const geometry::Geometry> geometry_ptr =
                                nullptr) override {
        PYBIND11_OVERLOAD(bool, VisualizerBase, UpdateGeometry, );
    }
    bool HasGeometry() const override {
        PYBIND11_OVERLOAD(bool, VisualizerBase, HasGeometry, );
    }
    void UpdateRender() override {
        PYBIND11_OVERLOAD(void, VisualizerBase, UpdateRender, );
    }
    void PrintVisualizerHelp() override {
        PYBIND11_OVERLOAD(void, VisualizerBase, PrintVisualizerHelp, );
    }
    void UpdateWindowTitle() override {
        PYBIND11_OVERLOAD(void, VisualizerBase, UpdateWindowTitle, );
    }
    void BuildUtilities() override {
        PYBIND11_OVERLOAD(void, VisualizerBase, BuildUtilities, );
    }
};

template <class ViewControlBase = visualization::ViewControl>
class PyViewControl : public ViewControlBase {
public:
    using ViewControlBase::ViewControlBase;
    void Reset() override { PYBIND11_OVERLOAD(void, ViewControlBase, Reset, ); }
    void ChangeFieldOfView(float step) override {
        PYBIND11_OVERLOAD(void, ViewControlBase, ChangeFieldOfView, step);
    }
    void ChangeWindowSize(int width, int height) override {
        PYBIND11_OVERLOAD(void, ViewControlBase, ChangeWindowSize, width,
                          height);
    }
    void Scale(float scale) override {
        PYBIND11_OVERLOAD(void, ViewControlBase, Scale, scale);
    }
    void Rotate(float x, float y, float xo, float yo) override {
        PYBIND11_OVERLOAD(void, ViewControlBase, Rotate, x, y, xo, yo);
    }
    void Translate(float x, float y, float xo, float yo) override {
        PYBIND11_OVERLOAD(void, ViewControlBase, Translate, x, y, xo, yo);
    }
};