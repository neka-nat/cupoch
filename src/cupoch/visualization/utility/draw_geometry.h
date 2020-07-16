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
///
/// \param geometry_list List of geometries to be visualized.
/// \param window_name The displayed title of the visualization window.
/// \param width The width of the visualization window.
/// \param height The height of the visualization window.
/// \param left margin of the visualization window.
/// \param top The top margin of the visualization window.
/// \param point_show_normal visualize point normals if set to true.
/// \param mesh_show_wireframe visualize mesh wireframe if set to true.
/// \param mesh_show_back_face visualize also the back face of the mesh
/// triangles.
bool DrawGeometries(const std::vector<std::shared_ptr<const geometry::Geometry>>
                            &geometry_ptrs,
                    const std::string &window_name = "Cupoch",
                    int width = 640,
                    int height = 480,
                    int left = 50,
                    int top = 50,
                    bool point_show_normal = false,
                    bool mesh_show_wireframe = false,
                    bool mesh_show_back_face = false);

}  // namespace visualization
}  // namespace cupoch