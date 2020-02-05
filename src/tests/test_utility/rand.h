#pragma once

#include <Eigen/Core>
#include <thrust/host_vector.h>

#include "cupoch/utility/eigen.h"

namespace unit_test {
// Initialize an Eigen::Vector3f.
// Output range: [vmin:vmax].
void Rand(Eigen::Vector3f& v,
          const Eigen::Vector3f& vmin,
          const Eigen::Vector3f& vmax,
          const int& seed);

// Initialize an Eigen::Vector3d.
// Output range: [vmin:vmax].
void Rand(Eigen::Vector3f& v,
          const float& vmin,
          const float& vmax,
          const int& seed);

// Initialize an Eigen::Vector2i vector.
// Output range: [vmin:vmax].
void Rand(thrust::host_vector<Eigen::Vector2i>& v,
          const Eigen::Vector2i& vmin,
          const Eigen::Vector2i& vmax,
          const int& seed);

// Initialize an Eigen::Vector3i vector.
// Output range: [vmin:vmax].
void Rand(thrust::host_vector<Eigen::Vector3i>& v,
          const Eigen::Vector3i& vmin,
          const Eigen::Vector3i& vmax,
          const int& seed);

// Initialize an Eigen::Vector3f vector.
// Output range: [vmin:vmax].
void Rand(thrust::host_vector<Eigen::Vector3f>& v,
          const Eigen::Vector3f& vmin,
          const Eigen::Vector3f& vmax,
          const int& seed);

// Initialize an Eigen::Vector4i vector.
// Output range: [vmin:vmax].
void Rand(thrust::host_vector<Eigen::Vector4i>& v,
          const int& vmin,
          const int& vmax,
          const int& seed);

// Initialize an Eigen::Vector4i vector.
// Output range: [vmin:vmax].
void Rand(thrust::host_vector<Eigen::Vector4i>& v,
          const Eigen::Vector4i& vmin,
          const Eigen::Vector4i& vmax,
          const int& seed);

// Initialize a uint8_t vector.
// Output range: [vmin:vmax].
void Rand(thrust::host_vector<uint8_t>& v,
          const uint8_t& vmin,
          const uint8_t& vmax,
          const int& seed);

// Initialize an array of int.
// Output range: [vmin:vmax].
void Rand(int* const v,
          const size_t& size,
          const int& vmin,
          const int& vmax,
          const int& seed);

// Initialize an int vector.
// Output range: [vmin:vmax].
void Rand(thrust::host_vector<int>& v,
          const int& vmin,
          const int& vmax,
          const int& seed);

// Initialize a size_t vector.
// Output range: [vmin:vmax].
void Rand(thrust::host_vector<size_t>& v,
          const size_t& vmin,
          const size_t& vmax,
          const int& seed);

// Initialize an array of float.
// Output range: [vmin:vmax].
void Rand(float* const v,
          const size_t& size,
          const float& vmin,
          const float& vmax,
          const int& seed);

// Initialize a float vector.
// Output range: [vmin:vmax].
void Rand(thrust::host_vector<float>& v,
          const float& vmin,
          const float& vmax,
          const int& seed);

}  // namespace unit_test