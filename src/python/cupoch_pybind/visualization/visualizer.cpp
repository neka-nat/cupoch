#include "cupoch/visualization/visualizer/visualizer.h"
#include "cupoch/geometry/image.h"

#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/visualization/visualization.h"
#include "cupoch_pybind/visualization/visualization_trampoline.h"

using namespace cupoch;

// Functions have similar arguments, thus the arg docstrings may be shared
static const std::unordered_map<std::string, std::string>
        map_visualizer_docstrings = {
                {"callback_func", "The call back function."},
                {"depth_scale",
                 "Scale depth value when capturing the depth image."},
                {"do_render", "Set to ``True`` to do render."},
                {"filename", "Path to file."},
                {"geometry", "The ``Geometry`` object."},
                {"height", "Height of window."},
                {"left", "Left margin of the window to the screen."},
                {"top", "Top margin of the window to the screen."},
                {"visible", "Whether the window is visible."},
                {"width", "Width of the window."},
                {"window_name", "Window title name."},
                {"convert_to_world_coordinate",
                 "Set to ``True`` to convert to world coordinates"},
                {"reset_bounding_box",
                 "Set to ``False`` to keep current viewpoint"}};

void pybind_visualizer(py::module &m) {
    py::class_<visualization::Visualizer, PyVisualizer<>,
               std::shared_ptr<visualization::Visualizer>>
            visualizer(m, "Visualizer", "The main Visualizer class.");
    py::detail::bind_default_constructor<visualization::Visualizer>(visualizer);
    visualizer
            .def("__repr__",
                 [](const visualization::Visualizer &vis) {
                     return std::string("Visualizer with name ") +
                            vis.GetWindowName();
                 })
            .def("create_window",
                 &visualization::Visualizer::CreateVisualizerWindow,
                 "Function to create a window and initialize GLFW",
                 "window_name"_a = "cupoch", "width"_a = 1920,
                 "height"_a = 1080, "left"_a = 50, "top"_a = 50,
                 "visible"_a = true)
            .def("destroy_window",
                 &visualization::Visualizer::DestroyVisualizerWindow,
                 "Function to destroy a window")
            .def("register_animation_callback",
                 &visualization::Visualizer::RegisterAnimationCallback,
                 "Function to register a callback function for animation",
                 "callback_func"_a)
            .def("run", &visualization::Visualizer::Run,
                 "Function to activate the window")
            .def("close", &visualization::Visualizer::Close,
                 "Function to notify the window to be closed")
            .def("reset_view_point", &visualization::Visualizer::ResetViewPoint,
                 "Function to reset view point")
            .def("update_geometry", &visualization::Visualizer::UpdateGeometry,
                 "Function to update geometry")
            .def("update_renderer", &visualization::Visualizer::UpdateRender,
                 "Function to inform render needed to be updated")
            .def("poll_events", &visualization::Visualizer::PollEvents,
                 "Function to poll events")
            .def("add_geometry", &visualization::Visualizer::AddGeometry,
                 "Function to add geometry to the scene and create "
                 "corresponding shaders",
                 "geometry"_a, "reset_bounding_box"_a = true)
            .def("remove_geometry", &visualization::Visualizer::RemoveGeometry,
                 "Function to remove geometry", "geometry"_a,
                 "reset_bounding_box"_a = true)
            .def("clear_geometries",
                 &visualization::Visualizer::ClearGeometries,
                 "Function to clear geometries from the visualizer")
            .def("get_view_control", &visualization::Visualizer::GetViewControl,
                 "Function to retrieve the associated ``ViewControl``",
                 py::return_value_policy::reference_internal)
            .def("get_render_option",
                 &visualization::Visualizer::GetRenderOption,
                 "Function to retrieve the associated ``RenderOption``",
                 py::return_value_policy::reference_internal)
            .def("capture_screen_image",
                 &visualization::Visualizer::CaptureScreenImage,
                 "Function to capture and save a screen image", "filename"_a,
                 "do_render"_a = false)
            .def("capture_depth_image",
                 &visualization::Visualizer::CaptureDepthImage,
                 "Function to capture and save a depth image", "filename"_a,
                 "do_render"_a = false, "depth_scale"_a = 1000.0)
            .def("get_window_name", &visualization::Visualizer::GetWindowName);

    docstring::ClassMethodDocInject(m, "Visualizer", "add_geometry",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "remove_geometry",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "capture_depth_image",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "capture_screen_image",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "close",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "create_window",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "destroy_window",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "get_render_option",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "get_view_control",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "get_window_name",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "poll_events",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer",
                                    "register_animation_callback",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "reset_view_point",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "run",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "update_geometry",
                                    map_visualizer_docstrings);
    docstring::ClassMethodDocInject(m, "Visualizer", "update_renderer",
                                    map_visualizer_docstrings);
}

void pybind_visualizer_method(py::module &m) {}