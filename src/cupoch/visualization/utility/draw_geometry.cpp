#include "cupoch/visualization/utility/draw_geometry.h"

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/visualization/visualizer/visualizer.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::visualization;

bool cupoch::visualization::DrawGeometries(const std::vector<std::shared_ptr<const geometry::Geometry>>
                                                   &geometry_ptrs,
                                           const std::string &window_name /* = "Cupoch"*/,
                                           int width /* = 640*/,
                                           int height /* = 480*/,
                                           int left /* = 50*/,
                                           int top /* = 50*/) {
    Visualizer visualizer;
    if (visualizer.CreateVisualizerWindow(window_name, width, height, left,
                                          top) == false) {
        utility::LogWarning("[DrawGeometries] Failed creating OpenGL window.");
        return false;
    }
    for (const auto &geometry_ptr : geometry_ptrs) {
        if (visualizer.AddGeometry(geometry_ptr) == false) {
            utility::LogWarning("[DrawGeometries] Failed adding geometry.");
            utility::LogWarning(
                    "[DrawGeometries] Possibly due to bad geometry or wrong "
                    "geometry type.");
            return false;
        }
    }
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    return true;
}