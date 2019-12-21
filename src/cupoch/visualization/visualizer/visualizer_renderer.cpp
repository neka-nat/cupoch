#include "cupoch/visualization/visualizer/visualizer.h"
#include "cupoch/geometry/trianglemesh.h"

using namespace cupoch;
using namespace cupoch::visualization;

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
            view_control_ptr_->FitInGeometry(*(geometry_ptr));
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