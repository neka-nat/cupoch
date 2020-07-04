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