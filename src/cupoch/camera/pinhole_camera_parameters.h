#pragma once

#include "cupoch/camera/pinhole_camera_intrinsic.h"

namespace cupoch {
namespace camera {

/// \class PinholeCameraParameters
///
/// \brief Contains both intrinsic and extrinsic pinhole camera parameters.
class PinholeCameraParameters : public utility::IJsonConvertible {
public:
    /// \brief Default Constructor.
    PinholeCameraParameters();
    ~PinholeCameraParameters() override;

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// PinholeCameraIntrinsic object.
    PinholeCameraIntrinsic intrinsic_;
    /// Camera extrinsic parameters.
    Eigen::Matrix4f_u extrinsic_;
};
}  // namespace camera
}  // namespace cupoch