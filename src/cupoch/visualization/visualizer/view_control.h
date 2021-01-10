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

#include "cupoch/camera/pinhole_camera_parameters.h"
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/geometry.h"
#include "cupoch/visualization/utility/gl_helper.h"
#include "cupoch/visualization/visualizer/view_parameters.h"

namespace cupoch {
namespace visualization {

class ViewControl {
public:
    static const float FIELD_OF_VIEW_MAX;
    static const float FIELD_OF_VIEW_MIN;
    static const float FIELD_OF_VIEW_DEFAULT;
    static const float FIELD_OF_VIEW_STEP;

    static const float ZOOM_DEFAULT;
    static const float ZOOM_MIN;
    static const float ZOOM_MAX;
    static const float ZOOM_STEP;

    static const float ROTATION_RADIAN_PER_PIXEL;

    enum ProjectionType {
        Perspective = 0,
        Orthogonal = 1,
    };

public:
    __host__ __device__ virtual ~ViewControl(){};

    /// Function to set view points
    /// This function obtains OpenGL context and calls OpenGL functions to set
    /// the view point.
    void SetViewMatrices(
            const Eigen::Matrix4f &model_matrix = Eigen::Matrix4f::Identity());

    /// Function to get equivalent view parameters (support orthogonal)
    bool ConvertToViewParameters(ViewParameters &status) const;
    bool ConvertFromViewParameters(const ViewParameters &status);

    /// Function to get equivalent pinhole camera parameters (does not support
    /// orthogonal since it is not a real camera view)
    bool ConvertToPinholeCameraParameters(
            camera::PinholeCameraParameters &parameters);
    bool ConvertFromPinholeCameraParameters(
            const camera::PinholeCameraParameters &parameters);

    ProjectionType GetProjectionType() const;
    void SetProjectionParameters();
    virtual void Reset();
    virtual void ChangeFieldOfView(float step);
    virtual void ChangeWindowSize(int width, int height);

    // Function to process scaling
    /// \param scale is the relative distance mouse has scrolled.
    virtual void Scale(float scale);

    // Function to process rotation
    /// \param x and \param y are the distances the mouse cursor has moved.
    /// \param xo and \param yo are the original point coordinate the mouse
    /// cursor started to move from.
    /// Coordinates are measured in screen coordinates relative to the top-left
    /// corner of the window client area.
    virtual void Rotate(float x, float y, float xo = 0.0, float yo = 0.0);

    // Function to process translation
    /// \param x and \param y are the distances the mouse cursor has moved.
    /// \param xo and \param yo are the original point coordinate the mouse
    /// cursor started to move from.
    /// Coordinates are measured in screen coordinates relative to the top-left
    /// corner of the window client area.
    virtual void Translate(float x, float y, float xo = 0.0, float yo = 0.0);

    // Function to process rolling
    /// \param x is the distances the mouse cursor has moved.
    /// Coordinates are measured in screen coordinates relative to the top-left
    /// corner of the window client area.
    virtual void Roll(float x);

    __host__ __device__ const geometry::AxisAlignedBoundingBox &GetBoundingBox()
            const {
        return bounding_box_;
    }

    void ResetBoundingBox() { bounding_box_.Clear(); }

    void FitInGeometry(const geometry::Geometry &geometry) {
        if (geometry.Dimension() == 3) {
            bounding_box_ += ((const geometry::GeometryBase3D &)geometry)
                                     .GetAxisAlignedBoundingBox();
        }
        SetProjectionParameters();
    }

    float GetFieldOfView() const { return field_of_view_; }
    gl_helper::GLMatrix4f GetMVPMatrix() const { return MVP_matrix_; }
    gl_helper::GLMatrix4f GetProjectionMatrix() const {
        return projection_matrix_;
    }
    gl_helper::GLMatrix4f GetViewMatrix() const { return view_matrix_; }
    gl_helper::GLMatrix4f GetModelMatrix() const { return model_matrix_; }
    gl_helper::GLVector3f GetEye() const { return eye_.cast<GLfloat>(); }
    gl_helper::GLVector3f GetLookat() const { return lookat_.cast<GLfloat>(); }
    gl_helper::GLVector3f GetUp() const { return up_.cast<GLfloat>(); }
    gl_helper::GLVector3f GetFront() const { return front_.cast<GLfloat>(); }
    gl_helper::GLVector3f GetRight() const { return right_.cast<GLfloat>(); }
    int GetWindowWidth() const { return window_width_; }
    int GetWindowHeight() const { return window_height_; }
    float GetZNear() const { return z_near_; }
    float GetZFar() const { return z_far_; }

    void SetConstantZNear(float z_near) { constant_z_near_ = z_near; }
    void SetConstantZFar(float z_far) { constant_z_far_ = z_far; }
    void UnsetConstantZNear() { constant_z_near_ = -1; }
    void UnsetConstantZFar() { constant_z_far_ = -1; }

protected:
    int window_width_ = 0;
    int window_height_ = 0;
    geometry::AxisAlignedBoundingBox bounding_box_;
    Eigen::Vector3f eye_;
    Eigen::Vector3f lookat_;
    Eigen::Vector3f up_;
    Eigen::Vector3f front_;
    Eigen::Vector3f right_;
    float distance_;
    float field_of_view_;
    float zoom_;
    float view_ratio_;
    float aspect_;
    float z_near_;
    float z_far_;
    float constant_z_near_ = -1;
    float constant_z_far_ = -1;
    gl_helper::GLMatrix4f projection_matrix_;
    gl_helper::GLMatrix4f view_matrix_;
    gl_helper::GLMatrix4f model_matrix_;
    gl_helper::GLMatrix4f MVP_matrix_;
};

}  // namespace visualization
}  // namespace cupoch