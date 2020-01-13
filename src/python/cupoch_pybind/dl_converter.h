#pragma once
#include <dlpack/dlpack.h>
#include <Eigen/Core>
#include <thrust/device_vector.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace cupoch {
namespace dlpack {

py::capsule ToDLpackCapsule(thrust::device_vector<Eigen::Vector3f>& src);

}
}