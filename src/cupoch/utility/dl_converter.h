#pragma once

#include <dlpack/dlpack.h>

#include <Eigen/Core>

#include "cupoch/utility/device_vector.h"

namespace cupoch {
namespace utility {

template <typename T, int Dim>
DLManagedTensor *ToDLPack(
        const utility::device_vector<Eigen::Matrix<T, Dim, 1>> &src);

template <typename T, int Dim>
void FromDLPack(const DLManagedTensor *src,
                utility::device_vector<Eigen::Matrix<T, Dim, 1>> &dst);

}  // namespace utility
}  // namespace cupoch