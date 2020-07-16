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
#include "cupoch/camera/pinhole_camera_parameters.h"

#include <json/json.h>

#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::camera;

PinholeCameraParameters::PinholeCameraParameters() {}

PinholeCameraParameters::~PinholeCameraParameters() {}

bool PinholeCameraParameters::ConvertToJsonValue(Json::Value &value) const {
    Json::Value trajectory_array;
    value["class_name"] = "PinholeCameraParameters";
    value["version_major"] = 1;
    value["version_minor"] = 0;
    if (EigenMatrix4fToJsonArray(extrinsic_, value["extrinsic"]) == false) {
        return false;
    }
    if (intrinsic_.ConvertToJsonValue(value["intrinsic"]) == false) {
        return false;
    }
    return true;
}

bool PinholeCameraParameters::ConvertFromJsonValue(const Json::Value &value) {
    if (value.isObject() == false) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "PinholeCameraParameters" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (intrinsic_.ConvertFromJsonValue(value["intrinsic"]) == false) {
        return false;
    }
    if (EigenMatrix4fFromJsonArray(extrinsic_, value["extrinsic"]) == false) {
        return false;
    }
    return true;
}