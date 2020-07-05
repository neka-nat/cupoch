#include "cupoch/geometry/laserscan.h"
#include "cupoch/geometry/boundingvolume.h"

#include "cupoch/utility/console.h"

namespace cupoch {
namespace geometry {

namespace {

std::pair<float, float> TangentMinMax(float min_angle, float max_angle) {
    float min_angle_tan = tan(min_angle);
    float max_angle_tan = tan(max_angle);
    // Correct sign of tan around singularity points
    if (min_angle_tan < 0.0)
        min_angle_tan = -min_angle_tan;
    if (max_angle_tan > 0.0)
        max_angle_tan = -max_angle_tan;
    return std::make_pair(min_angle_tan, max_angle_tan);
}

__device__
bool IsShadow(float r1, float r2, float included_angle, float min_angle_tan, float max_angle_tan) {
    const float perpendicular_y = r2 * sin(included_angle);
    const float perpendicular_x = r1 - r2 * cos(included_angle);
    const float perpendicular_tan = fabs(perpendicular_y) / perpendicular_x;

    if (perpendicular_tan > 0) {
        if (perpendicular_tan < min_angle_tan)
            return true;
    } else {
        if (perpendicular_tan > max_angle_tan)
            return true;
    }
    return false;
}

struct apply_scan_shadow_filter_functor {
    apply_scan_shadow_filter_functor(const float* ranges,
                                     float min_angle_tan,
                                     float max_angle_tan,
                                     float angle_increment,
                                     int num_steps,
                                     int num_scans,
                                     int window,
                                     int neighbors,
                                     bool remove_shadow_start_point,
                                     float* out)
                                     : ranges_(ranges),
                                     min_angle_tan_(min_angle_tan),
                                     max_angle_tan_(max_angle_tan),
                                     angle_increment_(angle_increment),
                                     num_steps_(num_steps), num_scans_(num_scans),
                                     window_(window), neighbors_(neighbors),
                                     remove_shadow_start_point_(remove_shadow_start_point),
                                     out_(out) {};
    const float* ranges_;
    const float min_angle_tan_;
    const float max_angle_tan_;
    const float angle_increment_;
    const int num_steps_;
    const int num_scans_;
    const int window_;
    const int neighbors_;
    const bool remove_shadow_start_point_;
    float* out_;
    __device__ void operator() (size_t idx) {
        int n = idx / num_steps_;
        int i = idx % num_steps_;
        for (int y = -window_; y < window_ + 1; y++) {
            int j = i + y;
            if (j < 0 || j >= num_steps_ || i == j) continue;
            if (IsShadow(ranges_[n * num_steps_ + i], ranges_[n * num_steps_ + j], y * angle_increment_,
                         min_angle_tan_, max_angle_tan_)) {
                for (int index = max(i - neighbors_, 0); index <= min(i + neighbors_, num_steps_ - 1); index++) {
                    if (ranges_[i] < ranges_[index]) {
                        out_[n * num_steps_ + index] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
                if (remove_shadow_start_point_) {
                    out_[n * num_steps_ + i] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
    }
};

}

LaserScan::LaserScan(int num_steps, float min_angle, float max_angle)
: GeometryBase<3>(Geometry::GeometryType::LaserScan), num_steps_(num_steps), min_angle_(min_angle), max_angle_(max_angle) {}

LaserScan::~LaserScan() {};

void LaserScan::SetRanges(const thrust::host_vector<float> &ranges) {
    ranges_ = ranges;
}

thrust::host_vector<float> LaserScan::GetRanges() const {
    thrust::host_vector<float> ranges = ranges_;
    return ranges;
}

void LaserScan::SetIntensities(const thrust::host_vector<float> &intensities) {
    intensities_ = intensities;
}

thrust::host_vector<float> LaserScan::GetIntensities() const {
    thrust::host_vector<float> intensities = intensities_;
    return intensities;
}

LaserScan &LaserScan::Clear() {
    num_scans_ = 0;
    ranges_.clear();
    intensities_.clear();
    return *this;
}

bool LaserScan::IsEmpty() const {
    return ranges_.empty();
}

Eigen::Vector3f LaserScan::GetMinBound() const {
    utility::LogError("LaserScan::GetMinBound is not supported");
    return Eigen::Vector3f::Zero();
}

Eigen::Vector3f LaserScan::GetMaxBound() const {
    utility::LogError("LaserScan::GetMaxBound is not supported");
    return Eigen::Vector3f::Zero();
}

Eigen::Vector3f LaserScan::GetCenter() const {
    utility::LogError("LaserScan::GetCenter is not supported");
    return Eigen::Vector3f::Zero();
}

AxisAlignedBoundingBox LaserScan::GetAxisAlignedBoundingBox() const {
    utility::LogError("LaserScan::GetAxisAlignedBoundingBox is not supported");
    return AxisAlignedBoundingBox();
}

LaserScan &LaserScan::Transform(const Eigen::Matrix4f &transformation) {
    origin_ = origin_ * transformation;
    return *this;
}

LaserScan &LaserScan::Translate(const Eigen::Vector3f &translation,
                                bool relative) {
    origin_.block<3, 1>(0, 3) += translation;
    return *this;
}

LaserScan &LaserScan::Scale(const float scale, bool center) {
    thrust::for_each(ranges_.begin(), ranges_.end(), [scale] __device__ (float &r) { r *= scale; });
    return *this;
}

LaserScan &LaserScan::Rotate(const Eigen::Matrix3f &R,
                             bool center) {
    origin_.block<3, 3>(0, 0) *= R;
    return *this;
}

std::shared_ptr<LaserScan> LaserScan::RangeFilter(float min_range, float max_range) const {
    auto out = std::make_shared<LaserScan>(num_steps_, min_angle_, max_angle_);
    if (max_range <= min_range) {
        utility::LogError("[RangeFilter] Invalid parameter with min_range greater than max_range.");
    }
    out->ranges_.resize(ranges_.size());
    thrust::transform(ranges_.begin(), ranges_.end(), out->ranges_.begin(),
                      [min_range, max_range] __device__ (float r) {
                          return (r < min_range || r > max_range) ? std::numeric_limits<float>::quiet_NaN() : r;
                      });
    return out;
}

std::shared_ptr<LaserScan> LaserScan::ScanShadowsFilter(float min_angle, float max_angle,
                                                        int window, int neighbors,
                                                        bool remove_shadow_start_point) const {
    auto out = std::make_shared<LaserScan>(num_steps_, min_angle_, max_angle_);
    *out = *this;
    auto minmax_tan = TangentMinMax(min_angle, max_angle);
    apply_scan_shadow_filter_functor func(thrust::raw_pointer_cast(ranges_.data()),
                                          minmax_tan.first, minmax_tan.second,
                                          GetAngleIncrement(),
                                          num_steps_, num_scans_,
                                          window, neighbors,
                                          remove_shadow_start_point,
                                          thrust::raw_pointer_cast(out->ranges_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(ranges_.size()), func);
    return out;
}

}
}