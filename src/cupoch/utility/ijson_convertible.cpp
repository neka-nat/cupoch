#include "cupoch/utility/ijson_convertible.h"

#include <json/json.h>

namespace cupoch {
namespace utility {

bool IJsonConvertible::EigenVector3fFromJsonArray(Eigen::Vector3f &vec,
                                                  const Json::Value &value) {
    if (value.size() != 3) {
        return false;
    } else {
        vec(0) = value[0].asFloat();
        vec(1) = value[1].asFloat();
        vec(2) = value[2].asFloat();
        return true;
    }
}

bool IJsonConvertible::EigenVector3fToJsonArray(const Eigen::Vector3f &vec,
                                                Json::Value &value) {
    value.clear();
    value.append(vec(0));
    value.append(vec(1));
    value.append(vec(2));
    return true;
}

bool IJsonConvertible::EigenVector4fFromJsonArray(Eigen::Vector4f &vec,
                                                  const Json::Value &value) {
    if (value.size() != 4) {
        return false;
    } else {
        vec(0) = value[0].asFloat();
        vec(1) = value[1].asFloat();
        vec(2) = value[2].asFloat();
        vec(3) = value[3].asFloat();
        return true;
    }
}

bool IJsonConvertible::EigenVector4fToJsonArray(const Eigen::Vector4f &vec,
                                                Json::Value &value) {
    value.clear();
    value.append(vec(0));
    value.append(vec(1));
    value.append(vec(2));
    value.append(vec(3));
    return true;
}

bool IJsonConvertible::EigenMatrix3fFromJsonArray(Eigen::Matrix3f &mat,
                                                  const Json::Value &value) {
    if (value.size() != 9) {
        return false;
    } else {
        for (int i = 0; i < 9; i++) {
            mat.coeffRef(i) = value[i].asFloat();
        }
        return true;
    }
}

bool IJsonConvertible::EigenMatrix3fToJsonArray(const Eigen::Matrix3f &mat,
                                                Json::Value &value) {
    value.clear();
    for (int i = 0; i < 9; i++) {
        value.append(mat.coeffRef(i));
    }
    return true;
}

bool IJsonConvertible::EigenMatrix4fFromJsonArray(Eigen::Matrix4f &mat,
                                                  const Json::Value &value) {
    if (value.size() != 16) {
        return false;
    } else {
        for (int i = 0; i < 16; i++) {
            mat.coeffRef(i) = value[i].asFloat();
        }
        return true;
    }
}

bool IJsonConvertible::EigenMatrix4fToJsonArray(const Eigen::Matrix4f &mat,
                                                Json::Value &value) {
    value.clear();
    for (int i = 0; i < 16; i++) {
        value.append(mat.coeffRef(i));
    }
    return true;
}

bool IJsonConvertible::EigenMatrix4fFromJsonArray(Eigen::Matrix4f_u &mat,
                                                  const Json::Value &value) {
    if (value.size() != 16) {
        return false;
    } else {
        for (int i = 0; i < 16; i++) {
            mat.coeffRef(i) = value[i].asFloat();
        }
        return true;
    }
}

bool IJsonConvertible::EigenMatrix4fToJsonArray(const Eigen::Matrix4f_u &mat,
                                                Json::Value &value) {
    value.clear();
    for (int i = 0; i < 16; i++) {
        value.append(mat.coeffRef(i));
    }
    return true;
}

bool IJsonConvertible::EigenMatrix6fFromJsonArray(Eigen::Matrix6f &mat,
                                                  const Json::Value &value) {
    if (value.size() != 36) {
        return false;
    } else {
        for (int i = 0; i < 36; i++) {
            mat.coeffRef(i) = value[i].asFloat();
        }
        return true;
    }
}

bool IJsonConvertible::EigenMatrix6fToJsonArray(const Eigen::Matrix6f &mat,
                                                Json::Value &value) {
    value.clear();
    for (int i = 0; i < 36; i++) {
        value.append(mat.coeffRef(i));
    }
    return true;
}

bool IJsonConvertible::EigenMatrix6fFromJsonArray(Eigen::Matrix6f_u &mat,
                                                  const Json::Value &value) {
    if (value.size() != 36) {
        return false;
    } else {
        for (int i = 0; i < 36; i++) {
            mat.coeffRef(i) = value[i].asFloat();
        }
        return true;
    }
}

bool IJsonConvertible::EigenMatrix6fToJsonArray(const Eigen::Matrix6f_u &mat,
                                                Json::Value &value) {
    value.clear();
    for (int i = 0; i < 36; i++) {
        value.append(mat.coeffRef(i));
    }
    return true;
}

}  // namespace utility
}  // namespace cupoch