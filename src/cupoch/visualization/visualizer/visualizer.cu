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
#include "cupoch/visualization/visualizer/visualizer.h"

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>

#include "cupoch/geometry/lineset.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/platform.h"
#include "cupoch/visualization/shader/geometry_renderer.h"

using namespace cupoch;
using namespace cupoch::visualization;

namespace {

class GLFWEnvironmentSingleton {
private:
    GLFWEnvironmentSingleton() { utility::LogDebug("GLFW init."); }
    GLFWEnvironmentSingleton(const GLFWEnvironmentSingleton &) = delete;
    GLFWEnvironmentSingleton &operator=(const GLFWEnvironmentSingleton &) =
            delete;

public:
    ~GLFWEnvironmentSingleton() {
        glfwTerminate();
        utility::LogDebug("GLFW destruct.");
    }

public:
    static GLFWEnvironmentSingleton &GetInstance() {
        static GLFWEnvironmentSingleton singleton;
        return singleton;
    }

    static int InitGLFW() {
        GLFWEnvironmentSingleton::GetInstance();
        return glfwInit();
    }

    static void GLFWErrorCallback(int error, const char *description) {
        utility::LogError("GLFW Error: {}", description);
    }
};

}  // unnamed namespace

Visualizer::Visualizer() {}

Visualizer::~Visualizer() {
    glfwTerminate();  // to be safe
}

bool Visualizer::CreateVisualizerWindow(
        const std::string &window_name /* = "Cupoch"*/,
        const int width /* = 640*/,
        const int height /* = 480*/,
        const int left /* = 50*/,
        const int top /* = 50*/,
        const bool visible /* = true*/) {
    window_name_ = window_name;
    glfwSetErrorCallback(GLFWEnvironmentSingleton::GLFWErrorCallback);
    if (!GLFWEnvironmentSingleton::InitGLFW()) {
        utility::LogWarning("Failed to initialize GLFW");
        return false;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#ifndef HEADLESS_RENDERING
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, visible ? 1 : 0);

    window_ = glfwCreateWindow(width, height, window_name_.c_str(), NULL, NULL);
    if (!window_) {
        utility::LogWarning("Failed to create window");
        return false;
    }
    glfwSetWindowPos(window_, left, top);
    glfwSetWindowUserPointer(window_, this);

#ifdef __APPLE__
    // Some hacks to get pixel_to_screen_coordinate_
    glfwSetWindowSize(window_, 100, 100);
    glfwSetWindowPos(window_, 100, 100);
    int pixel_width_in_osx, pixel_height_in_osx;
    glfwGetFramebufferSize(window_, &pixel_width_in_osx, &pixel_height_in_osx);
    if (pixel_width_in_osx > 0) {
        pixel_to_screen_coordinate_ = 100.0 / (double)pixel_width_in_osx;
    } else {
        pixel_to_screen_coordinate_ = 1.0;
    }
    glfwSetWindowSize(window_, std::round(width * pixel_to_screen_coordinate_),
                      std::round(height * pixel_to_screen_coordinate_));
    glfwSetWindowPos(window_, std::round(left * pixel_to_screen_coordinate_),
                     std::round(top * pixel_to_screen_coordinate_));
#endif  //__APPLE__

    auto window_refresh_callback = [](GLFWwindow *window) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->WindowRefreshCallback(window);
    };
    glfwSetWindowRefreshCallback(window_, window_refresh_callback);

    auto window_resize_callback = [](GLFWwindow *window, int w, int h) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->WindowResizeCallback(window, w, h);
    };
    glfwSetFramebufferSizeCallback(window_, window_resize_callback);

    auto mouse_move_callback = [](GLFWwindow *window, double x, double y) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->MouseMoveCallback(window, x, y);
    };
    glfwSetCursorPosCallback(window_, mouse_move_callback);

    auto mouse_scroll_callback = [](GLFWwindow *window, double x, double y) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->MouseScrollCallback(window, x, y);
    };
    glfwSetScrollCallback(window_, mouse_scroll_callback);

    auto mouse_button_callback = [](GLFWwindow *window, int button, int action,
                                    int mods) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->MouseButtonCallback(window, button, action, mods);
    };
    glfwSetMouseButtonCallback(window_, mouse_button_callback);

    auto key_press_callback = [](GLFWwindow *window, int key, int scancode,
                                 int action, int mods) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->KeyPressCallback(window, key, scancode, action, mods);
    };
    glfwSetKeyCallback(window_, key_press_callback);

    auto window_close_callback = [](GLFWwindow *window) {
        static_cast<Visualizer *>(glfwGetWindowUserPointer(window))
                ->WindowCloseCallback(window);
    };
    glfwSetWindowCloseCallback(window_, window_close_callback);

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    if (InitOpenGL() == false) {
        return false;
    }

    if (InitViewControl() == false) {
        return false;
    }

    if (InitRenderOption() == false) {
        return false;
    }

    int window_width, window_height;
    glfwGetFramebufferSize(window_, &window_width, &window_height);
    WindowResizeCallback(window_, window_width, window_height);

    UpdateWindowTitle();

    const char *glsl_version = "#version 130";
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    ImGui::StyleColorsDark();
    // Setup ImGui binding
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    is_initialized_ = true;
    return true;
}

void Visualizer::DestroyVisualizerWindow() {
    is_initialized_ = false;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glDeleteVertexArrays(1, &vao_id_);
    glfwDestroyWindow(window_);
}

void Visualizer::RegisterAnimationCallback(
        std::function<bool(Visualizer *)> callback_func) {
    animation_callback_func_ = callback_func;
}

bool Visualizer::InitViewControl() {
    view_control_ptr_ = std::unique_ptr<ViewControl>(new ViewControl);
    ResetViewPoint();
    return true;
}

bool Visualizer::InitRenderOption() {
    render_option_ptr_ = std::unique_ptr<RenderOption>(new RenderOption);
    return true;
}

void Visualizer::UpdateWindowTitle() {
    if (window_ != NULL) {
        glfwSetWindowTitle(window_, window_name_.c_str());
    }
}

void Visualizer::BuildUtilities() {
    glfwMakeContextCurrent(window_);

    // 0. Build coordinate frame
    const auto boundingbox = GetViewControl().GetBoundingBox();
    coordinate_frame_mesh_ptr_ = geometry::TriangleMesh::CreateCoordinateFrame(
            boundingbox.GetMaxExtent() * 0.2);
    coordinate_frame_mesh_renderer_ptr_ =
            std::make_shared<glsl::CoordinateFrameRenderer>();
    // 1. Build grid line
    grid_line_ptr_ = geometry::LineSet<3>::CreateSquareGrid();
    grid_line_renderer_ptr_ = std::make_shared<glsl::GridLineRenderer>();
    if (coordinate_frame_mesh_renderer_ptr_->AddGeometry(
                coordinate_frame_mesh_ptr_) == false) {
        return;
    }
    if (grid_line_renderer_ptr_->AddGeometry(
                grid_line_ptr_) == false) {
        return;
    }
    utility_ptrs_.emplace_back(coordinate_frame_mesh_ptr_);
    utility_renderer_ptrs_.emplace_back(coordinate_frame_mesh_renderer_ptr_);
    utility_ptrs_.emplace_back(grid_line_ptr_);
    utility_renderer_ptrs_.emplace_back(grid_line_renderer_ptr_);
}

void Visualizer::Run() {
    BuildUtilities();
    UpdateWindowTitle();
    while (bool(animation_callback_func_) ? PollEvents() : WaitEvents()) {
        if (bool(animation_callback_func_in_loop_)) {
            if (animation_callback_func_in_loop_(this)) {
                UpdateGeometry();
            }
            // Set render flag as dirty anyways, because when we use callback
            // functions, we assume something has been changed in the callback
            // and the redraw event should be triggered.
            UpdateRender();
        }
        RenderImGui();
    }
}

void Visualizer::RenderImGui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    bool status_changed = false;
    {
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin("Infomation");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::Text("Optional rendering");
        status_changed |= ImGui::Checkbox("Origin", &render_option_ptr_->show_coordinate_frame_);
        status_changed |= ImGui::Checkbox("Grid", &render_option_ptr_->show_grid_line_);
        ImGui::Text("Visible");
        int count = 0;
        for (auto &geometry_ptr : geometry_ptrs_) {
            status_changed |= ImGui::Checkbox(
                    ("Geometry " + std::to_string(count)).c_str(),
                    &geometry_ptr.second);
            ++count;
        }
        ImGui::End();
    }
    if (status_changed) UpdateRender();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window_);
}

void Visualizer::Close() {
    glfwSetWindowShouldClose(window_, GL_TRUE);
    utility::LogDebug("[Visualizer] Window closing.");
}

bool Visualizer::WaitEvents() {
    if (!is_initialized_) {
        return false;
    }
    glfwMakeContextCurrent(window_);
    if (is_redraw_required_) {
        WindowRefreshCallback(window_);
    }
    animation_callback_func_in_loop_ = animation_callback_func_;
    glfwWaitEvents();
    return !glfwWindowShouldClose(window_);
}

bool Visualizer::PollEvents() {
    if (!is_initialized_) {
        return false;
    }
    glfwMakeContextCurrent(window_);
    if (is_redraw_required_) {
        WindowRefreshCallback(window_);
    }
    animation_callback_func_in_loop_ = animation_callback_func_;
    glfwPollEvents();
    return !glfwWindowShouldClose(window_);
}

bool Visualizer::AddGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr,
        bool reset_bounding_box) {
    if (!is_initialized_) {
        return false;
    }
    glfwMakeContextCurrent(window_);
    std::shared_ptr<glsl::GeometryRenderer> renderer_ptr;
    if (geometry_ptr->GetGeometryType() ==
        geometry::Geometry::GeometryType::Unspecified) {
        return false;
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::PointCloud) {
        renderer_ptr = std::make_shared<glsl::PointCloudRenderer>();
        if (!renderer_ptr->AddGeometry(geometry_ptr)) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::VoxelGrid) {
        renderer_ptr = std::make_shared<glsl::VoxelGridRenderer>();
        if (!renderer_ptr->AddGeometry(geometry_ptr)) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::OccupancyGrid) {
        renderer_ptr = std::make_shared<glsl::OccupancyGridRenderer>();
        if (!renderer_ptr->AddGeometry(geometry_ptr)) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::DistanceTransform) {
        renderer_ptr = std::make_shared<glsl::DistanceTransformRenderer>();
        if (!renderer_ptr->AddGeometry(geometry_ptr)) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::LineSet) {
        renderer_ptr = std::make_shared<glsl::LineSetRenderer>();
        if (!renderer_ptr->AddGeometry(geometry_ptr)) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::Graph &&
               geometry_ptr->Dimension() == 2) {
        renderer_ptr = std::make_shared<glsl::GraphRenderer<2>>();
        if (!renderer_ptr->AddGeometry(geometry_ptr)) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::Graph &&
               geometry_ptr->Dimension() == 3) {
        renderer_ptr = std::make_shared<glsl::GraphRenderer<3>>();
        if (!renderer_ptr->AddGeometry(geometry_ptr)) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::TriangleMesh) {
        renderer_ptr = std::make_shared<glsl::TriangleMeshRenderer>();
        if (!renderer_ptr->AddGeometry(geometry_ptr)) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::Image) {
        renderer_ptr = std::make_shared<glsl::ImageRenderer>();
        if (!renderer_ptr->AddGeometry(geometry_ptr)) {
            return false;
        }
    } else {
        return false;
    }
    geometry_renderer_ptrs_.insert(renderer_ptr);
    geometry_ptrs_[geometry_ptr] = true;
    if (reset_bounding_box) {
        view_control_ptr_->FitInGeometry(*geometry_ptr);
        ResetViewPoint();
    }
    utility::LogDebug(
            "Add geometry and update bounding box to {}",
            view_control_ptr_->GetBoundingBox().GetPrintInfo().c_str());
    return UpdateGeometry();
}

bool Visualizer::RemoveGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr,
        bool reset_bounding_box) {
    if (is_initialized_ == false) {
        return false;
    }
    glfwMakeContextCurrent(window_);
    std::shared_ptr<glsl::GeometryRenderer> geometry_renderer_delete = NULL;
    for (auto &geometry_renderer_ptr : geometry_renderer_ptrs_) {
        if (geometry_renderer_ptr->GetGeometry() == geometry_ptr)
            geometry_renderer_delete = geometry_renderer_ptr;
    }
    if (geometry_renderer_delete == NULL) return false;
    geometry_renderer_ptrs_.erase(geometry_renderer_delete);
    geometry_ptrs_.erase(geometry_ptr);
    if (reset_bounding_box) {
        ResetViewPoint(true);
    }
    utility::LogDebug(
            "Remove geometry and update bounding box to {}",
            view_control_ptr_->GetBoundingBox().GetPrintInfo().c_str());
    return UpdateGeometry();
}

bool Visualizer::ClearGeometries() {
    if (is_initialized_ == false) {
        return false;
    }
    glfwMakeContextCurrent(window_);
    geometry_renderer_ptrs_.clear();
    geometry_ptrs_.clear();
    return UpdateGeometry();
}

bool Visualizer::UpdateGeometry(
        std::shared_ptr<const geometry::Geometry> geometry_ptr) {
    glfwMakeContextCurrent(window_);
    bool success = true;
    for (const auto &renderer_ptr : geometry_renderer_ptrs_) {
        if (geometry_ptr == nullptr ||
            renderer_ptr->HasGeometry(geometry_ptr)) {
            success = (success && renderer_ptr->UpdateGeometry());
        }
    }
    UpdateRender();
    return success;
}

void Visualizer::UpdateRender() { is_redraw_required_ = true; }

bool Visualizer::HasGeometry() const { return !geometry_ptrs_.empty(); }

void Visualizer::PrintVisualizerHelp() {
    // clang-format off
    utility::LogInfo("  -- Mouse view control --");
    utility::LogInfo("    Left button + drag         : Rotate.");
    utility::LogInfo("    Ctrl + left button + drag  : Translate.");
    utility::LogInfo("    Wheel button + drag        : Translate.");
    utility::LogInfo("    Shift + left button + drag : Roll.");
    utility::LogInfo("    Wheel                      : Zoom in/out.");
    utility::LogInfo("");
    utility::LogInfo("  -- Keyboard view control --");
    utility::LogInfo("    [/]          : Increase/decrease field of view.");
    utility::LogInfo("    R            : Reset view point.");
    utility::LogInfo("    Ctrl/Cmd + C : Copy current view status into the clipboard.");
    utility::LogInfo("    Ctrl/Cmd + V : Paste view status from clipboard.");
    utility::LogInfo("");
    utility::LogInfo("  -- General control --");
    utility::LogInfo("    Q, Esc       : Exit window.");
    utility::LogInfo("    H            : Print help message.");
    utility::LogInfo("    P, PrtScn    : Take a screen capture.");
    utility::LogInfo("    D            : Take a depth capture.");
    utility::LogInfo("    O            : Take a capture of current rendering settings.");
    utility::LogInfo("");
    utility::LogInfo("  -- Render mode control --");
    utility::LogInfo("    L            : Turn on/off lighting.");
    utility::LogInfo("    +/-          : Increase/decrease point size.");
    utility::LogInfo("    Ctrl + +/-   : Increase/decrease width of geometry::LineSet.");
    utility::LogInfo("    N            : Turn on/off point cloud normal rendering.");
    utility::LogInfo("    S            : Toggle between mesh flat shading and smooth shading.");
    utility::LogInfo("    W            : Turn on/off mesh wireframe.");
    utility::LogInfo("    B            : Turn on/off back face rendering.");
    utility::LogInfo("    I            : Turn on/off image zoom in interpolation.");
    utility::LogInfo("    T            : Toggle among image render:");
    utility::LogInfo("                   no stretch / keep ratio / freely stretch.");
    utility::LogInfo("");
    utility::LogInfo("  -- Color control --");
    utility::LogInfo("    0..4,9       : Set point cloud color option.");
    utility::LogInfo("                   0 - Default behavior, render point color.");
    utility::LogInfo("                   1 - Render point color.");
    utility::LogInfo("                   2 - x coordinate as color.");
    utility::LogInfo("                   3 - y coordinate as color.");
    utility::LogInfo("                   4 - z coordinate as color.");
    utility::LogInfo("                   9 - normal as color.");
    utility::LogInfo("    Ctrl + 0..4,9: Set mesh color option.");
    utility::LogInfo("                   0 - Default behavior, render uniform gray color.");
    utility::LogInfo("                   1 - Render point color.");
    utility::LogInfo("                   2 - x coordinate as color.");
    utility::LogInfo("                   3 - y coordinate as color.");
    utility::LogInfo("                   4 - z coordinate as color.");
    utility::LogInfo("                   9 - normal as color.");
    utility::LogInfo("    Shift + 0..4 : Color map options.");
    utility::LogInfo("                   0 - Gray scale color.");
    utility::LogInfo("                   1 - JET color map.");
    utility::LogInfo("                   2 - SUMMER color map.");
    utility::LogInfo("                   3 - WINTER color map.");
    utility::LogInfo("                   4 - HOT color map.");
    utility::LogInfo("");
}