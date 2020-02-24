#pragma once

#include "cupoch/registration/transformation_estimation.h"
#include "cupoch/utility/device_vector.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace registration {

Eigen::Matrix4f_u Kabsch(const utility::device_vector<Eigen::Vector3f> &model,
                         const utility::device_vector<Eigen::Vector3f> &target,
                         const CorrespondenceSet &corres);

Eigen::Matrix4f_u Kabsch(const utility::device_vector<Eigen::Vector3f> &model,
                         const utility::device_vector<Eigen::Vector3f> &target);

}  // namespace registration
}  // namespace cupoch