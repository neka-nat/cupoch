/**
 * Copyright (c) 2023 Neka-Nat
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
#include "cupoch/geometry/laserscanbuffer.h"
#include "cupoch/geometry/image.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/platform.h"
#include <Eigen/Geometry>
#include <thrust/tabulate.h>

namespace cupoch {
namespace geometry {

namespace {

struct pointcloud_to_laserscan_functor {
    pointcloud_to_laserscan_functor(
        float* ranges,
        const float min_height,
        const float height_increment,
        const int num_steps,
        const float min_range,
        const float max_range,
        const float min_angle,
        const float max_angle,
        const float angle_increment)
        : ranges_(ranges),
          min_height_(min_height),
          height_increment_(height_increment),
          num_steps_(num_steps),
          min_range_(min_range),
          max_range_(max_range),
          min_angle_(min_angle),
          max_angle_(max_angle),
          angle_increment_(angle_increment) {}

    float* ranges_;
    const float min_height_;
    const float height_increment_;
    const int num_steps_;
    const float min_range_;
    const float max_range_;
    const float min_angle_;
    const float max_angle_;
    const float angle_increment_;

    __device__
    void operator() (const Eigen::Vector3f& pt) const {
        const float z = pt.z();
        const int jndex = (z - min_height_) / height_increment_;
        const float range = hypotf(pt.x(), pt.y());
        if (range < min_range_ || range > max_range_) {
            return;
        }
        const float angle = atan2f(pt.y(), pt.x());
        if (angle < min_angle_ || angle > max_angle_) {
            return;
        }
        const int index = (angle - min_angle_) / angle_increment_;
        const float range_old = ranges_[num_steps_ * jndex + index];
        if (isnan(range_old) || range < range_old) {
            atomicExch(ranges_ + num_steps_ * jndex + index, range);
        }
    }
};

}

std::shared_ptr<LaserScanBuffer> LaserScanBuffer::CreateFromPointCloud(
        const geometry::PointCloud &pcd,
        float angle_increment,
        float min_height,
        float max_height,
        unsigned int num_vertical_divisions,
        float min_range,
        float max_range,
        float min_angle,
        float max_angle) {
    if (angle_increment <= 0.0) {
        utility::LogError("[LaserScanBuffer::CreateFromPointCloud] angle_increment must be positive.");
        return std::shared_ptr<LaserScanBuffer>();
    }
    if (min_height >= max_height) {
        utility::LogError("[LaserScanBuffer::CreateFromPointCloud] min_height must be smaller than max_height.");
        return std::shared_ptr<LaserScanBuffer>();
    }
    size_t num_steps = std::ceil((max_angle - min_angle) / angle_increment);
    size_t num_max_scans = std::max(DEFAULT_NUM_MAX_SCANS, num_vertical_divisions);
    auto laserscanbuffer = std::make_shared<LaserScanBuffer>(
        num_steps,
        num_max_scans,
        min_angle,
        max_angle);
    laserscanbuffer->ranges_.resize(num_steps * num_max_scans, std::numeric_limits<float>::quiet_NaN());
    laserscanbuffer->origins_.resize(num_max_scans);
    thrust::tabulate(
        laserscanbuffer->origins_.begin(),
        laserscanbuffer->origins_.end(),
        [min_height, max_height, num_vertical_divisions] __device__ (int i) {
            Eigen::Matrix4f origin = Eigen::Matrix4f::Identity();
            origin(2, 3) = min_height + (max_height - min_height) * i / num_vertical_divisions;
            return origin;
        });
    auto func = pointcloud_to_laserscan_functor(
        thrust::raw_pointer_cast(laserscanbuffer->ranges_.data()),
        min_height,
        (max_height - min_height) / num_vertical_divisions,
        num_steps,
        min_range,
        max_range,
        min_angle,
        max_angle,
        angle_increment);
    thrust::for_each(pcd.points_.begin(), pcd.points_.end(), func);
    return laserscanbuffer;
}


std::shared_ptr<LaserScanBuffer> LaserScanBuffer::CreateFromDepthImage(
        const geometry::Image &depth,
        const camera::PinholeCameraIntrinsic &intrinsic,
        float angle_increment,
        float min_y,
        float max_y,
        unsigned int num_vertical_divisions,
        float min_range,
        float max_range,
        float min_angle,
        float max_angle,
        float depth_scale,
        float depth_trunc,
        int stride) {
    Eigen::Matrix4f x_rot90 = Eigen::Matrix4f::Identity();
    x_rot90.block<3, 3>(0, 0) = Eigen::AngleAxisf(-M_PI_2, Eigen::Vector3f::UnitX()).toRotationMatrix();
    auto pcd = geometry::PointCloud::CreateFromDepthImage(
        depth, intrinsic, x_rot90, depth_scale, depth_trunc, stride);
    auto laserscanbuffer = CreateFromPointCloud(
        *pcd,
        angle_increment,
        min_y,
        max_y,
        num_vertical_divisions,
        min_range,
        max_range,
        min_angle,
        max_angle);
    thrust::for_each(
        laserscanbuffer->origins_.begin(),
        laserscanbuffer->origins_.end(),
        [x_rot90] __device__ (Eigen::Matrix4f_u& origin) {
            origin = x_rot90 * origin;
        });
    return laserscanbuffer;
}

}
}