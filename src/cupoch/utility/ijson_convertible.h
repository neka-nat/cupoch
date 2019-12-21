#pragma once

#include <Eigen/Core>

#include "cupoch/utility/eigen.h"

namespace Json {
class Value;
}  // namespace Json

namespace cupoch {
namespace utility {

/// Class IJsonConvertible defines the behavior of a class that can convert
/// itself to/from a json::Value.
class IJsonConvertible {
public:
    virtual ~IJsonConvertible() {}

public:
    virtual bool ConvertToJsonValue(Json::Value &value) const = 0;
    virtual bool ConvertFromJsonValue(const Json::Value &value) = 0;

public:
    static bool EigenVector3fFromJsonArray(Eigen::Vector3f &vec,
                                           const Json::Value &value);
    static bool EigenVector3fToJsonArray(const Eigen::Vector3f &vec,
                                         Json::Value &value);
    static bool EigenVector4fFromJsonArray(Eigen::Vector4f &vec,
                                           const Json::Value &value);
    static bool EigenVector4fToJsonArray(const Eigen::Vector4f &vec,
                                         Json::Value &value);
    static bool EigenMatrix3fFromJsonArray(Eigen::Matrix3f &mat,
                                           const Json::Value &value);
    static bool EigenMatrix3fToJsonArray(const Eigen::Matrix3f &mat,
                                         Json::Value &value);
    static bool EigenMatrix4fFromJsonArray(Eigen::Matrix4f &mat,
                                           const Json::Value &value);
    static bool EigenMatrix4fToJsonArray(const Eigen::Matrix4f &mat,
                                         Json::Value &value);
    static bool EigenMatrix4fFromJsonArray(Eigen::Matrix4f_u &mat,
                                           const Json::Value &value);
    static bool EigenMatrix4fToJsonArray(const Eigen::Matrix4f_u &mat,
                                         Json::Value &value);
    static bool EigenMatrix6fFromJsonArray(Eigen::Matrix6f &mat,
                                           const Json::Value &value);
    static bool EigenMatrix6fToJsonArray(const Eigen::Matrix6f &mat,
                                         Json::Value &value);
    static bool EigenMatrix6fFromJsonArray(Eigen::Matrix6f_u &mat,
                                           const Json::Value &value);
    static bool EigenMatrix6fToJsonArray(const Eigen::Matrix6f_u &mat,
                                         Json::Value &value);
};

}  // namespace utility
}  // namespace cupoch