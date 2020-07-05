#pragma once

#include "cupoch/geometry/geometry_base.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace geometry {

class AxisAlignedBoundingBox;

class LaserScan : public GeometryBase<3> {
public:
    LaserScan(int num_steps, float min_angle, float max_angle);
    ~LaserScan();

    void SetRanges(const thrust::host_vector<float> &ranges);
    thrust::host_vector<float> GetRanges() const;

    void SetIntensities(const thrust::host_vector<float> &intensities);
    thrust::host_vector<float> GetIntensities() const;

    LaserScan &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3f GetMinBound() const override;
    Eigen::Vector3f GetMaxBound() const override;
    Eigen::Vector3f GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    LaserScan &Transform(const Eigen::Matrix4f &transformation) override;
    LaserScan &Translate(const Eigen::Vector3f &translation,
                         bool relative = true) override;
    LaserScan &Scale(const float scale, bool center = true) override;
    LaserScan &Rotate(const Eigen::Matrix3f &R,
                      bool center = true) override;

    bool HasIntensities() const { return !intensities_.empty(); };
    float GetAngleIncrement() const { return (max_angle_ - min_angle_) / (num_steps_ - 1); };
    std::shared_ptr<LaserScan> RangeFilter(float min_range, float max_range) const;
    std::shared_ptr<LaserScan> ScanShadowsFilter(float min_angle,
                                                 float max_angle,
                                                 int window,
                                                 int neighbors = 0,
                                                 bool remove_shadow_start_point = false) const;

public:
    utility::device_vector<float> ranges_;
    utility::device_vector<float> intensities_;
    int num_scans_;
    int num_steps_;
    float min_angle_;
    float max_angle_;
    Eigen::Matrix4f_u origin_ = Eigen::Matrix4f_u::Identity();
};

}
}