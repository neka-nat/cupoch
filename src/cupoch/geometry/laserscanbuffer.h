#pragma once

#include "cupoch/geometry/geometry_base.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace geometry {

class AxisAlignedBoundingBox;

class LaserScanBuffer : public GeometryBase<3> {
public:
    LaserScanBuffer(int num_steps, int num_max_scans = 10, float min_angle = -M_PI, float max_angle = M_PI);
    ~LaserScanBuffer();
    LaserScanBuffer(const LaserScanBuffer& other);

    thrust::host_vector<float> GetRanges() const;
    thrust::host_vector<float> GetIntensities() const;

    LaserScanBuffer &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3f GetMinBound() const override;
    Eigen::Vector3f GetMaxBound() const override;
    Eigen::Vector3f GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    LaserScanBuffer &Transform(const Eigen::Matrix4f &transformation) override;
    LaserScanBuffer &Translate(const Eigen::Vector3f &translation,
                               bool relative = true) override;
    LaserScanBuffer &Scale(const float scale, bool center = true) override;
    LaserScanBuffer &Rotate(const Eigen::Matrix3f &R,
                            bool center = true) override;

    bool HasIntensities() const { return !intensities_.empty(); };
    float GetAngleIncrement() const { return (max_angle_ - min_angle_) / (num_steps_ - 1); };

    LaserScanBuffer &AddRanges(const utility::device_vector<float>& ranges,
                               const Eigen::Matrix4f& transformation = Eigen::Matrix4f::Identity(),
                               const utility::device_vector<float>& intensities = utility::device_vector<float>());
    LaserScanBuffer &AddRanges(const thrust::host_vector<float>& ranges,
                               const Eigen::Matrix4f& transformation = Eigen::Matrix4f::Identity(),
                               const thrust::host_vector<float>& intensities = thrust::host_vector<float>());

    std::shared_ptr<LaserScanBuffer> RangeFilter(float min_range, float max_range) const;
    std::shared_ptr<LaserScanBuffer> ScanShadowsFilter(float min_angle,
                                                 float max_angle,
                                                 int window,
                                                 int neighbors = 0,
                                                 bool remove_shadow_start_point = false) const;

public:
    utility::device_vector<float> ranges_;
    utility::device_vector<float> intensities_;
    int top_ = 0;
    int bottom_ = 0;
    const int num_steps_;
    const int num_max_scans_;
    float min_angle_;
    float max_angle_;
    utility::device_vector<Eigen::Matrix4f_u> origins_;
};

}
}