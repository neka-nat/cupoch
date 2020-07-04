#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/io/class_io/ijson_convertible_io.h"
#include "cupoch/utility/filesystem.h"
#include "cupoch/visualization/utility/draw_geometry.h"
#include "cupoch/visualization/visualizer/visualizer.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/visualization/visualization.h"

using namespace cupoch;

// Visualization util functions have similar arguments, sharing arg docstrings
static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"callback_function",
                 "Call back function to be triggered at a key press event."},
                {"filename", "The file path."},
                {"geometry_list", "List of geometries to be visualized."},
                {"height", "The height of the visualization window."},
                {"key_to_callback", "Map of key to call back functions."},
                {"left", "The left margin of the visualization window."},
                {"optional_view_trajectory_json_file",
                 "Camera trajectory json file path for custom animation."},
                {"top", "The top margin of the visualization window."},
                {"width", "The width of the visualization window."},
                {"window_name",
                 "The displayed title of the visualization window."}};

void pybind_visualization_utility_methods(py::module &m) {
    m.def(
            "draw_geometries",
            [](const std::vector<std::shared_ptr<const geometry::Geometry>>
                       &geometry_ptrs,
               const std::string &window_name, int width, int height, int left,
               int top, bool point_show_normal, bool mesh_show_wireframe,
               bool mesh_show_back_face) {
                std::string current_dir =
                        utility::filesystem::GetWorkingDirectory();
                visualization::DrawGeometries(
                        geometry_ptrs, window_name, width, height, left, top,
                        point_show_normal, mesh_show_wireframe,
                        mesh_show_back_face);
                utility::filesystem::ChangeWorkingDirectory(current_dir);
            },
            "Function to draw a list of geometry::Geometry objects",
            "geometry_list"_a, "window_name"_a = "cupoch", "width"_a = 1920,
            "height"_a = 1080, "left"_a = 50, "top"_a = 50,
            "point_show_normal"_a = false, "mesh_show_wireframe"_a = false,
            "mesh_show_back_face"_a = false);
    docstring::FunctionDocInject(m, "draw_geometries",
                                 map_shared_argument_docstrings);
}