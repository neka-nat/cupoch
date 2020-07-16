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
#include "cupoch/camera/pinhole_camera_intrinsic.h"

#include <json/json.h>

#include <Eigen/Dense>

#include "cupoch/utility/console.h"

namespace cupoch {
namespace camera {

PinholeCameraIntrinsic::PinholeCameraIntrinsic()
    : width_(-1), height_(-1), intrinsic_matrix_(Eigen::Matrix3f::Zero()) {}

PinholeCameraIntrinsic::PinholeCameraIntrinsic(
        int width, int height, float fx, float fy, float cx, float cy) {
    SetIntrinsics(width, height, fx, fy, cx, cy);
}

PinholeCameraIntrinsic::PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters param) {
    if (param == PinholeCameraIntrinsicParameters::PrimeSenseDefault)
        SetIntrinsics(640, 480, 525.0, 525.0, 319.5, 239.5);
    else if (param ==
             PinholeCameraIntrinsicParameters::Kinect2DepthCameraDefault)
        SetIntrinsics(512, 424, 365.456, 365.456, 254.878, 205.395);
    else if (param ==
             PinholeCameraIntrinsicParameters::Kinect2ColorCameraDefault)
        SetIntrinsics(1920, 1080, 1059.9718, 1059.9718, 975.7193, 545.9533);
}

PinholeCameraIntrinsic::~PinholeCameraIntrinsic() {}

bool PinholeCameraIntrinsic::ConvertToJsonValue(Json::Value &value) const {
    value["width"] = width_;
    value["height"] = height_;
    if (EigenMatrix3fToJsonArray(intrinsic_matrix_,
                                 value["intrinsic_matrix"]) == false) {
        return false;
    }
    return true;
}

bool PinholeCameraIntrinsic::ConvertFromJsonValue(const Json::Value &value) {
    if (value.isObject() == false) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.");
        return false;
    }
    width_ = value.get("width", -1).asInt();
    height_ = value.get("height", -1).asInt();
    if (EigenMatrix3fFromJsonArray(intrinsic_matrix_,
                                   value["intrinsic_matrix"]) == false) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: wrong format.");
        return false;
    }
    return true;
}
}  // namespace camera
}  // namespace cupoch