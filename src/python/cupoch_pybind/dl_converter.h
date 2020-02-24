#pragma once
#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>

#include "cupoch/utility/device_vector.h"
namespace py = pybind11;

namespace cupoch {
namespace dlpack {

py::capsule ToDLpackCapsule(utility::device_vector<Eigen::Vector3f>& src);

void FromDLpackCapsule(py::capsule dlpack,
                       utility::device_vector<Eigen::Vector3f>& dst);

}  // namespace dlpack
}  // namespace cupoch