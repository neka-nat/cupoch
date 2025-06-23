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

#include "cupoch/geometry/geometry_base.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {

namespace camera {
class PinholeCameraIntrinsic;
}  // namespace camera

namespace geometry {

template <int Dim>
class AxisAlignedBoundingBox;
class Image;
class PointCloud;


static const unsigned int DEFAULT_NUM_MAX_SCANS = 50;

class LaserScanBuffer : public GeometryBase3D {
public:
    LaserScanBuffer(int num_steps,
                    int num_max_scans = DEFAULT_NUM_MAX_SCANS,
                    float min_angle = -M_PI,
                    float max_angle = M_PI);
    ~LaserScanBuffer();
    LaserScanBuffer(const LaserScanBuffer &other);

    std::vector<float> GetRanges() const;
    std::vector<float> GetIntensities() const;

    LaserScanBuffer &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3f GetMinBound() const override;
    Eigen::Vector3f GetMaxBound() const override;
    Eigen::Vector3f GetCenter() const override;
    AxisAlignedBoundingBox<3> GetAxisAlignedBoundingBox() const override;
    LaserScanBuffer &Transform(const Eigen::Matrix4f &transformation) override;
    LaserScanBuffer &Translate(const Eigen::Vector3f &translation,
                               bool relative = true) override;
    LaserScanBuffer &Scale(const float scale, bool center = true) override;
    LaserScanBuffer &Rotate(const Eigen::Matrix3f &R,
                            bool center = true) override;

    bool HasIntensities() const { return !intensities_.empty(); };
    float GetAngleIncrement() const {
        return (max_angle_ - min_angle_) / (num_steps_ - 1);
    };

    bool IsFull() const { return GetNumScans() == num_max_scans_; };
    int GetNumScans() const { return bottom_ - top_; };

    template <typename ContainerType>
    LaserScanBuffer &AddRanges(
            const ContainerType &ranges,
            const Eigen::Matrix4f &transformation = Eigen::Matrix4f::Identity(),
            const ContainerType &intensities =
                    ContainerType());

    LaserScanBuffer &AddRanges(
            const float *ranges,
            const Eigen::Matrix4f &transformation = Eigen::Matrix4f::Identity(),
            const float *intensities = nullptr);

    LaserScanBuffer &Merge(const LaserScanBuffer &other);

    std::shared_ptr<LaserScanBuffer> PopOneScan();
    std::pair<std::unique_ptr<utility::pinned_host_vector<float>>, std::unique_ptr<utility::pinned_host_vector<float>>> PopHostOneScan();

    std::shared_ptr<LaserScanBuffer> RangeFilter(float min_range,
                                                 float max_range) const;
    std::shared_ptr<LaserScanBuffer> ScanShadowsFilter(
            float min_angle,
            float max_angle,
            int window,
            int neighbors = 0,
            bool remove_shadow_start_point = false) const;

    static std::shared_ptr<LaserScanBuffer> CreateFromPointCloud(
            const geometry::PointCloud &pcd,
            float angle_increment,
            float min_height,
            float max_height,
            unsigned int num_vertical_divisions = 1,
            float min_range = 0.0,
            float max_range = std::numeric_limits<float>::infinity(),
            float min_angle = -M_PI,
            float max_angle = M_PI);

    static std::shared_ptr<LaserScanBuffer> CreateFromDepthImage(
            const geometry::Image &depth,
            const camera::PinholeCameraIntrinsic &intrinsic,
            float angle_increment,
            float min_y,
            float max_y,
            unsigned int num_vertical_divisions = 1,
            float min_range = 0.0,
            float max_range = std::numeric_limits<float>::infinity(),
            float min_angle = -M_PI,
            float max_angle = M_PI,
            float depth_scale = 1000.0,
            float depth_trunc = 1000.0,
            int stride = 1);

public:
    utility::device_vector<float> ranges_;
    utility::device_vector<float> intensities_;
    int top_ = 0;     //!< index of the oldest scan
    int bottom_ = 0;  //!< index of the newest scan
    const int num_steps_;      //!< number of steps in a scan
    const int num_max_scans_;  //!< maximum number of scans
    float min_angle_;
    float max_angle_;
    utility::device_vector<Eigen::Matrix4f_u> origins_;
};

}  // namespace geometry
}  // namespace cupoch