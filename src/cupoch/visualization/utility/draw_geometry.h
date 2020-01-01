#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "cupoch/geometry/geometry.h"

namespace cupoch {
namespace visualization {

class Visualizer;

/// The convenient function of drawing something
/// This function is a wrapper that calls the core functions of Visualizer.
/// This function MUST be called from the main thread. It blocks the main thread
/// until the window is closed.

bool DrawGeometries(const std::vector<std::shared_ptr<const geometry::Geometry>>
                            &geometry_ptrs,
                    const std::string &window_name = "Cupoch",
                    int width = 640,
                    int height = 480,
                    int left = 50,
                    int top = 50);

}
}