#include "cupoch/visualization/visualizer/visualizer.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/utility/platform.h"
#include "cupoch/utility/console.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

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
        const std::string &window_name /* = "Open3D"*/,
        const int width /* = 640*/,
        const int height /* = 480*/,
        const int left /* = 50*/,
        const int top /* = 50*/,
        const bool visible /* = true*/) {
    cudaSafeCall(cudaGLSetGLDevice(utility::GetDevice()));
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

    return true;
}

void Visualizer::DestroyVisualizerWindow() {
    is_initialized_ = false;
    glDeleteVertexArrays(1, &vao_id_);
    glfwDestroyWindow(window_);
}

void Visualizer::RegisterAnimationCallback(
        std::function<bool(Visualizer *)> callback_func) {
    animation_callback_func_ = callback_func;
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
            boundingbox.GetMaxExtent() * 0.2, boundingbox.min_bound_);
    coordinate_frame_mesh_renderer_ptr_ =
            std::make_shared<glsl::CoordinateFrameRenderer>();
    if (coordinate_frame_mesh_renderer_ptr_->AddGeometry(
                coordinate_frame_mesh_ptr_) == false) {
        return;
    }
    utility_ptrs_.push_back(coordinate_frame_mesh_ptr_);
    utility_renderer_ptrs_.push_back(coordinate_frame_mesh_renderer_ptr_);
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
    }
}

void Visualizer::Close() {
    glfwSetWindowShouldClose(window_, GL_TRUE);
    utility::LogDebug("[Visualizer] Window closing.");
}

bool Visualizer::WaitEvents() {
    if (is_initialized_ == false) {
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
    if (is_initialized_ == false) {
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
    if (is_initialized_ == false) {
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
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
                       geometry::Geometry::GeometryType::TriangleMesh) {
        renderer_ptr = std::make_shared<glsl::TriangleMeshRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else if (geometry_ptr->GetGeometryType() ==
               geometry::Geometry::GeometryType::Image) {
        renderer_ptr = std::make_shared<glsl::ImageRenderer>();
        if (renderer_ptr->AddGeometry(geometry_ptr) == false) {
            return false;
        }
    } else {
        return false;
    }
    geometry_renderer_ptrs_.insert(renderer_ptr);
    geometry_ptrs_.insert(geometry_ptr);
    if (reset_bounding_box) {
        view_control_ptr_->FitInGeometry(*geometry_ptr);
        ResetViewPoint();
    }
    utility::LogDebug(
            "Add geometry and update bounding box to {}",
            view_control_ptr_->GetBoundingBox().GetPrintInfo().c_str());
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

void Visualizer::UpdateRender() {is_redraw_required_ = true;}

bool Visualizer::HasGeometry() const {return !geometry_ptrs_.empty();}