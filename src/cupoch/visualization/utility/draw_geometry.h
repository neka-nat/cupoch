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