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

#include "cupoch/registration/transformation_estimation.h"
#include "cupoch/utility/device_vector.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace registration {

Eigen::Matrix4f_u Kabsch(const utility::device_vector<Eigen::Vector3f> &model,
                         const utility::device_vector<Eigen::Vector3f> &target,
                         const CorrespondenceSet &corres);
Eigen::Matrix4f_u Kabsch(cudaStream_t stream1,
                         cudaStream_t stream2,
                         const utility::device_vector<Eigen::Vector3f> &model,
                         const utility::device_vector<Eigen::Vector3f> &target,
                         const CorrespondenceSet &corres);

Eigen::Matrix4f_u Kabsch(const utility::device_vector<Eigen::Vector3f> &model,
                         const utility::device_vector<Eigen::Vector3f> &target);
Eigen::Matrix4f_u Kabsch(cudaStream_t stream1,
                         cudaStream_t stream2,
                         const utility::device_vector<Eigen::Vector3f> &model,
                         const utility::device_vector<Eigen::Vector3f> &target);

Eigen::Matrix4f_u Kabsch(const std::vector<Eigen::Vector3f> &model,
                         const std::vector<Eigen::Vector3f> &target);


Eigen::Matrix4f_u KabschWeighted(
        const utility::device_vector<Eigen::Vector3f> &model,
        const utility::device_vector<Eigen::Vector3f> &target,
        const utility::device_vector<float> &weight);

}  // namespace registration
}  // namespace cupoch