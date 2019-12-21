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