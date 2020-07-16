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