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
#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>

#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/io/class_io/ijson_convertible_io.h"
#include "cupoch/io/class_io/image_io.h"
#include "cupoch/utility/console.h"
#include "cupoch/visualization/shader/geometry_renderer.h"
#include "cupoch/visualization/visualizer/view_trajectory.h"
#include "cupoch/visualization/visualizer/visualizer.h"

using namespace cupoch;
using namespace cupoch::visualization;

bool Visualizer::InitOpenGL() {
    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
        utility::LogWarning("Failed to initialize GLEW.");
        return false;
    }

    glGenVertexArrays(1, &vao_id_);
    glBindVertexArray(vao_id_);

    // depth test
    glEnable(GL_DEPTH_TEST);
    glClearDepth(1.0f);

    // pixel alignment
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // polygon rendering
    glEnable(GL_CULL_FACE);

    // glReadPixels always read front buffer
    glReadBuffer(GL_FRONT);

    return true;
}

void Visualizer::Render() {
    glfwMakeContextCurrent(window_);

    view_control_ptr_->SetViewMatrices();

    glEnable(GL_MULTISAMPLE);
    glDisable(GL_BLEND);
    auto &background_color = render_option_ptr_->background_color_;
    glClearColor((GLclampf)background_color(0), (GLclampf)background_color(1),
                 (GLclampf)background_color(2), 1.0f);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (const auto &renderer_ptr : geometry_renderer_ptrs_) {
        if (geometry_ptrs_[renderer_ptr->GetGeometry()])
            renderer_ptr->Render(*render_option_ptr_, *view_control_ptr_);
    }
    for (const auto &renderer_ptr : utility_renderer_ptrs_) {
        RenderOption *opt = render_option_ptr_.get();
        auto optIt = utility_renderer_opts_.find(renderer_ptr);
        if (optIt != utility_renderer_opts_.end()) {
            opt = &optIt->second;
        }
        renderer_ptr->Render(*opt, *view_control_ptr_);
    }

    glfwSwapBuffers(window_);
}

void Visualizer::ResetViewPoint(bool reset_bounding_box /* = false*/) {
    if (reset_bounding_box) {
        view_control_ptr_->ResetBoundingBox();
        for (const auto &geometry_ptr : geometry_ptrs_) {
            view_control_ptr_->FitInGeometry(*(geometry_ptr.first));
        }
        if (coordinate_frame_mesh_ptr_ && coordinate_frame_mesh_renderer_ptr_) {
            const auto &boundingbox = view_control_ptr_->GetBoundingBox();
            coordinate_frame_mesh_ptr_ =
                    geometry::TriangleMesh::CreateCoordinateFrame(
                            boundingbox.GetMaxExtent() * 0.2,
                            boundingbox.min_bound_);
            coordinate_frame_mesh_renderer_ptr_->UpdateGeometry();
        }
    }
    view_control_ptr_->Reset();
    is_redraw_required_ = true;
}

void Visualizer::CopyViewStatusToClipboard() {
    ViewParameters current_status;
    if (view_control_ptr_->ConvertToViewParameters(current_status) == false) {
        utility::LogError("Something is wrong copying view status.");
    }
    ViewTrajectory trajectory;
    trajectory.view_status_.emplace_back(current_status);
    std::string clipboard_string;
    if (io::WriteIJsonConvertibleToJSONString(clipboard_string, trajectory) ==
        false) {
        utility::LogError("Something is wrong copying view status.");
    }
    glfwSetClipboardString(window_, clipboard_string.c_str());
}

void Visualizer::CopyViewStatusFromClipboard() {
    const char *clipboard_string_buffer = glfwGetClipboardString(window_);
    if (clipboard_string_buffer != NULL) {
        std::string clipboard_string(clipboard_string_buffer);
        ViewTrajectory trajectory;
        if (io::ReadIJsonConvertibleFromJSONString(clipboard_string,
                                                   trajectory) == false) {
            utility::LogError("Something is wrong copying view status.");
        }
        if (trajectory.view_status_.size() != 1) {
            utility::LogError("Something is wrong copying view status.");
        }
        view_control_ptr_->ConvertFromViewParameters(
                trajectory.view_status_[0]);
    }
}

void Visualizer::CaptureScreenImage(const std::string &filename /* = ""*/,
                                    bool do_render /* = true*/) {
    std::string png_filename = filename;
    std::string camera_filename;
    if (png_filename.empty()) {
        std::string timestamp = utility::GetCurrentTimeStamp();
        png_filename = "ScreenCapture_" + timestamp + ".png";
        camera_filename = "ScreenCamera_" + timestamp + ".json";
    }
    geometry::Image screen_image;
    screen_image.Prepare(view_control_ptr_->GetWindowWidth(),
                         view_control_ptr_->GetWindowHeight(), 3, 1);
    if (do_render) {
        Render();
        is_redraw_required_ = false;
    }
    glFinish();
    auto host_data = screen_image.GetData();
    glReadPixels(0, 0, view_control_ptr_->GetWindowWidth(),
                 view_control_ptr_->GetWindowHeight(), GL_RGB, GL_UNSIGNED_BYTE,
                 host_data.data());

    // glReadPixels get the screen in a vertically flipped manner
    // Thus we should flip it back.
    io::HostImage png_image;
    png_image.Prepare(view_control_ptr_->GetWindowWidth(),
                      view_control_ptr_->GetWindowHeight(), 3, 1);
    int bytes_per_line = screen_image.BytesPerLine();
    for (int i = 0; i < screen_image.height_; i++) {
        memcpy(thrust::raw_pointer_cast(png_image.data_.data()) + bytes_per_line * i,
               thrust::raw_pointer_cast(host_data.data()) +
                       bytes_per_line * (screen_image.height_ - i - 1),
               bytes_per_line);
    }

    utility::LogDebug("[Visualizer] Screen capture to {}",
                      png_filename.c_str());
    io::WriteImage(png_filename, png_image);
    if (!camera_filename.empty()) {
        utility::LogDebug("[Visualizer] Screen camera capture to {}",
                          camera_filename.c_str());
        camera::PinholeCameraParameters parameter;
        view_control_ptr_->ConvertToPinholeCameraParameters(parameter);
        io::WriteIJsonConvertible(camera_filename, parameter);
    }
}

std::shared_ptr<geometry::Image> Visualizer::CaptureDepthFloatBuffer(
        bool do_render /* = true*/) {
    geometry::Image depth_image;
    depth_image.Prepare(view_control_ptr_->GetWindowWidth(),
                        view_control_ptr_->GetWindowHeight(), 1, 4);
    if (do_render) {
        Render();
        is_redraw_required_ = false;
    }
    glFinish();

    auto host_data = depth_image.GetData();
#if __APPLE__
    // On OSX with Retina display and glfw3, there is a bug with glReadPixels().
    // When using glReadPixels() to read a block of depth data. The data is
    // horizontally stretched (vertically it is fine). This issue is related
    // to GLFW_SAMPLES hint. When it is set to 0 (anti-aliasing disabled),
    // glReadPixels() works fine. See this post for details:
    // http://stackoverflow.com/questions/30608121/glreadpixel-one-pass-vs-looping-through-points
    // The reason of this bug is unknown. The current workaround is to read
    // depth buffer column by column. This is 15~30 times slower than one block
    // reading glReadPixels().
    std::vector<float> float_buffer(depth_image.height_);
    float *p = (float *)host_data.data();
    for (int j = 0; j < depth_image.width_; j++) {
        glReadPixels(j, 0, 1, depth_image.height_, GL_DEPTH_COMPONENT, GL_FLOAT,
                     float_buffer.data());
        for (int i = 0; i < depth_image.height_; i++) {
            p[i * depth_image.width_ + j] = float_buffer[i];
        }
    }
#else   //__APPLE__
    // By default, glReadPixels read a block of depth buffer.
    glReadPixels(0, 0, depth_image.width_, depth_image.height_,
                 GL_DEPTH_COMPONENT, GL_FLOAT, host_data.data());
#endif  //__APPLE__

    // glReadPixels get the screen in a vertically flipped manner
    // We should flip it back, and convert it to the correct depth value
    auto image_ptr = std::make_shared<io::HostImage>();
    double z_near = view_control_ptr_->GetZNear();
    double z_far = view_control_ptr_->GetZFar();

    image_ptr->Prepare(view_control_ptr_->GetWindowWidth(),
                       view_control_ptr_->GetWindowHeight(), 1, 4);
    for (int i = 0; i < depth_image.height_; i++) {
        float *p_depth = (float *)(thrust::raw_pointer_cast(host_data.data()) +
                                   depth_image.BytesPerLine() *
                                           (depth_image.height_ - i - 1));
        float *p_image = (float *)(thrust::raw_pointer_cast(image_ptr->data_.data()) +
                                   image_ptr->BytesPerLine() * i);
        for (int j = 0; j < depth_image.width_; j++) {
            if (p_depth[j] == 1.0) {
                continue;
            }
            double z_depth =
                    2.0 * z_near * z_far /
                    (z_far + z_near -
                     (2.0 * (double)p_depth[j] - 1.0) * (z_far - z_near));
            p_image[j] = (float)z_depth;
        }
    }
    auto output = std::make_shared<geometry::Image>();
    image_ptr->ToDevice(*output);
    return output;
}

void Visualizer::CaptureDepthImage(const std::string &filename /* = ""*/,
                                   bool do_render /* = true*/,
                                   double depth_scale /* = 1000.0*/) {
    std::string png_filename = filename;
    std::string camera_filename;
    if (png_filename.empty()) {
        std::string timestamp = utility::GetCurrentTimeStamp();
        png_filename = "DepthCapture_" + timestamp + ".png";
        camera_filename = "DepthCamera_" + timestamp + ".json";
    }
    geometry::Image depth_image;
    depth_image.Prepare(view_control_ptr_->GetWindowWidth(),
                        view_control_ptr_->GetWindowHeight(), 1, 4);

    if (do_render) {
        Render();
        is_redraw_required_ = false;
    }
    glFinish();

    auto host_data = depth_image.GetData();
#if __APPLE__
    // On OSX with Retina display and glfw3, there is a bug with glReadPixels().
    // When using glReadPixels() to read a block of depth data. The data is
    // horizontally streched (vertically it is fine). This issue is related
    // to GLFW_SAMPLES hint. When it is set to 0 (anti-aliasing disabled),
    // glReadPixels() works fine. See this post for details:
    // http://stackoverflow.com/questions/30608121/glreadpixel-one-pass-vs-looping-through-points
    // The reason of this bug is unknown. The current workaround is to read
    // depth buffer column by column. This is 15~30 times slower than one block
    // reading glReadPixels().
    std::vector<float> float_buffer(depth_image.height_);
    float *p = (float *)host_data.data();
    for (int j = 0; j < depth_image.width_; j++) {
        glReadPixels(j, 0, 1, depth_image.width_, GL_DEPTH_COMPONENT, GL_FLOAT,
                     float_buffer.data());
        for (int i = 0; i < depth_image.height_; i++) {
            p[i * depth_image.width_ + j] = float_buffer[i];
        }
    }
#else   //__APPLE__
    // By default, glReadPixels read a block of depth buffer.
    glReadPixels(0, 0, depth_image.width_, depth_image.height_,
                 GL_DEPTH_COMPONENT, GL_FLOAT, host_data.data());
#endif  //__APPLE__

    // glReadPixels get the screen in a vertically flipped manner
    // We should flip it back, and convert it to the correct depth value
    io::HostImage png_image;
    double z_near = view_control_ptr_->GetZNear();
    double z_far = view_control_ptr_->GetZFar();

    png_image.Prepare(view_control_ptr_->GetWindowWidth(),
                      view_control_ptr_->GetWindowHeight(), 1, 2);
    for (int i = 0; i < depth_image.height_; i++) {
        float *p_depth = (float *)(thrust::raw_pointer_cast(host_data.data()) +
                                   depth_image.BytesPerLine() *
                                           (depth_image.height_ - i - 1));
        uint16_t *p_png = (uint16_t *)(thrust::raw_pointer_cast(png_image.data_.data()) +
                                       png_image.BytesPerLine() * i);
        for (int j = 0; j < depth_image.width_; j++) {
            if (p_depth[j] == 1.0) {
                continue;
            }
            double z_depth =
                    2.0 * z_near * z_far /
                    (z_far + z_near -
                     (2.0 * (double)p_depth[j] - 1.0) * (z_far - z_near));
            p_png[j] = (uint16_t)std::min(std::round(depth_scale * z_depth),
                                          (double)INT16_MAX);
        }
    }

    utility::LogDebug("[Visualizer] Depth capture to {}", png_filename.c_str());
    io::WriteImage(png_filename, png_image);
    if (!camera_filename.empty()) {
        utility::LogDebug("[Visualizer] Depth camera capture to {}",
                          camera_filename.c_str());
        camera::PinholeCameraParameters parameter;
        view_control_ptr_->ConvertToPinholeCameraParameters(parameter);
        io::WriteIJsonConvertible(camera_filename, parameter);
    }
}

void Visualizer::CaptureRenderOption(const std::string &filename /* = ""*/) {
    std::string json_filename = filename;
    if (json_filename.empty()) {
        std::string timestamp = utility::GetCurrentTimeStamp();
        json_filename = "RenderOption_" + timestamp + ".json";
    }
    utility::LogDebug("[Visualizer] Render option capture to {}",
                      json_filename.c_str());
    io::WriteIJsonConvertible(json_filename, *render_option_ptr_);
}