#include "cupoch/geometry/laserscanbuffer.h"
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
                                     int window,
                                     int neighbors,
                                     bool remove_shadow_start_point,
                                     float* out)
                                     : ranges_(ranges),
                                     min_angle_tan_(min_angle_tan),
                                     max_angle_tan_(max_angle_tan),
                                     angle_increment_(angle_increment),
                                     num_steps_(num_steps),
                                     window_(window), neighbors_(neighbors),
                                     remove_shadow_start_point_(remove_shadow_start_point),
                                     out_(out) {};
    const float* ranges_;
    const float min_angle_tan_;
    const float max_angle_tan_;
    const float angle_increment_;
    const int num_steps_;
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

LaserScanBuffer::LaserScanBuffer(int num_steps, int num_max_scans, float min_angle, float max_angle)
: GeometryBase<3>(Geometry::GeometryType::LaserScanBuffer),
num_steps_(num_steps), num_max_scans_(num_max_scans),
min_angle_(min_angle), max_angle_(max_angle) {}

LaserScanBuffer::~LaserScanBuffer() {};

LaserScanBuffer::LaserScanBuffer(const LaserScanBuffer& other)
: GeometryBase<3>(Geometry::GeometryType::LaserScanBuffer),
ranges_(other.ranges_), intensities_(other.intensities_),
top_(other.top_), bottom_(other.bottom_),
num_steps_(other.num_steps_), num_max_scans_(other.num_max_scans_),
min_angle_(other.min_angle_), max_angle_(other.max_angle_),
origins_(other.origins_) {}

thrust::host_vector<float> LaserScanBuffer::GetRanges() const {
    thrust::host_vector<float> ranges;
    if (top_ == bottom_) {
        return ranges;
    }
    int start = top_ % num_max_scans_;
    int end = bottom_ % num_max_scans_;
    if (start < end) {
        int n = end - start;
        ranges.resize(n * num_steps_);
        thrust::copy_n(ranges_.begin() + start * num_steps_, n * num_steps_, ranges.begin());
        return ranges;
    } else {
        ranges.resize(num_max_scans_ * num_steps_);
        int offset = (num_max_scans_ - start) * num_steps_;
        thrust::copy_n(ranges_.begin() + start * num_steps_, offset, ranges.begin());
        thrust::copy_n(ranges_.begin(), end * num_steps_, ranges.begin() + offset);
        return ranges;
    }
}

thrust::host_vector<float> LaserScanBuffer::GetIntensities() const {
    thrust::host_vector<float> intensities;
    if (top_ == bottom_) {
        return intensities;
    }
    int start = top_ % num_max_scans_;
    int end = bottom_ % num_max_scans_;
    if (start < end) {
        int n = start - end;
        intensities.resize(n * num_steps_);
        thrust::copy_n(intensities_.begin() + start * num_steps_, n * num_steps_, intensities.begin());
        return intensities;
    } else {
        intensities.resize(num_max_scans_ * num_steps_);
        int offset = (num_max_scans_ - start) * num_steps_;
        thrust::copy_n(intensities_.begin() + start * num_steps_, offset, intensities.begin());
        thrust::copy_n(intensities_.begin(), end * num_steps_, intensities.begin() + offset);
        return intensities;
    }
}

LaserScanBuffer &LaserScanBuffer::Clear() {
    top_ = 0;
    bottom_ = 0;
    ranges_.clear();
    intensities_.clear();
    return *this;
}

bool LaserScanBuffer::IsEmpty() const {
    return ranges_.empty();
}

Eigen::Vector3f LaserScanBuffer::GetMinBound() const {
    utility::LogError("LaserScanBuffer::GetMinBound is not supported");
    return Eigen::Vector3f::Zero();
}

Eigen::Vector3f LaserScanBuffer::GetMaxBound() const {
    utility::LogError("LaserScanBuffer::GetMaxBound is not supported");
    return Eigen::Vector3f::Zero();
}

Eigen::Vector3f LaserScanBuffer::GetCenter() const {
    utility::LogError("LaserScanBuffer::GetCenter is not supported");
    return Eigen::Vector3f::Zero();
}

AxisAlignedBoundingBox LaserScanBuffer::GetAxisAlignedBoundingBox() const {
    utility::LogError("LaserScanBuffer::GetAxisAlignedBoundingBox is not supported");
    return AxisAlignedBoundingBox();
}

LaserScanBuffer &LaserScanBuffer::Transform(const Eigen::Matrix4f &transformation) {
    thrust::for_each(origins_.begin(), origins_.end(),
                     [transformation] __device__ (Eigen::Matrix4f_u& trans) {
                         trans = trans * transformation;
                     });
    return *this;
}

LaserScanBuffer &LaserScanBuffer::Translate(const Eigen::Vector3f &translation,
                                bool relative) {
    thrust::for_each(origins_.begin(), origins_.end(),
                     [translation] __device__ (Eigen::Matrix4f_u& trans) {
                        trans.block<3, 1>(0, 3) = trans.block<3, 1>(0, 3) + translation;
                     });
    return *this;
}

LaserScanBuffer &LaserScanBuffer::Scale(const float scale, bool center) {
    thrust::for_each(ranges_.begin(), ranges_.end(), [scale] __device__ (float &r) { r *= scale; });
    return *this;
}

LaserScanBuffer &LaserScanBuffer::Rotate(const Eigen::Matrix3f &R,
                                         bool center) {
    thrust::for_each(origins_.begin(), origins_.end(),
                     [R] __device__ (Eigen::Matrix4f_u& trans) {
                        trans.block<3, 3>(0, 0) = trans.block<3, 3>(0, 0) * R;
                     });
    return *this;
}

LaserScanBuffer &LaserScanBuffer::AddRanges(const utility::device_vector<float>& ranges,
                                            const Eigen::Matrix4f& transformation,
                                            const utility::device_vector<float>& intensities) {
    if (ranges.size() != num_steps_) {
        utility::LogError("[AddRanges] Invalid size of input ranges.");
        return *this;
    }
    if (HasIntensities() && ranges.size() != intensities.size()) {
        utility::LogError("[AddRanges] Invalid size of intensities.");
        return *this;
    }

    bool add_intensities = !intensities.empty() && ranges.size() == intensities.size();
    int end = bottom_ % num_max_scans_;
    if (bottom_ + 1 <= num_max_scans_) {
        ranges_.insert(ranges_.end(), ranges.begin(), ranges.end());
        if (add_intensities) intensities_.insert(intensities_.end(), intensities.begin(), intensities.end());
        origins_.push_back(transformation);
        bottom_++;
    } else {
        thrust::copy_n(ranges.begin(), num_steps_, ranges_.begin() + end * num_steps_);
        if (add_intensities) thrust::copy_n(intensities.begin(), num_steps_, intensities_.begin() + end * num_steps_);
        origins_[end] = transformation;
        top_++;
        bottom_++;
    }
    return *this;
}

LaserScanBuffer &LaserScanBuffer::AddRanges(const thrust::host_vector<float>& ranges,
                                            const Eigen::Matrix4f& transformation,
                                            const thrust::host_vector<float>& intensities) {
    utility::device_vector<float> d_ranges = ranges;
    utility::device_vector<float> d_intensities = intensities;
    return AddRanges(d_ranges, transformation, d_intensities);
}

std::shared_ptr<LaserScanBuffer> LaserScanBuffer::RangeFilter(float min_range, float max_range) const {
    auto out = std::make_shared<LaserScanBuffer>(num_steps_, num_max_scans_, min_angle_, max_angle_);
    if (max_range <= min_range) {
        utility::LogError("[RangeFilter] Invalid parameter with min_range greater than max_range.");
    }
    out->ranges_.resize(ranges_.size());
    out->top_ = top_;
    out->bottom_ = bottom_;
    thrust::transform(ranges_.begin(), ranges_.end(), out->ranges_.begin(),
                      [min_range, max_range] __device__ (float r) {
                          return (r < min_range || r > max_range) ? std::numeric_limits<float>::quiet_NaN() : r;
                      });
    return out;
}

std::shared_ptr<LaserScanBuffer> LaserScanBuffer::ScanShadowsFilter(float min_angle, float max_angle,
                                                                    int window, int neighbors,
                                                                    bool remove_shadow_start_point) const {
    auto out = std::make_shared<LaserScanBuffer>(*this);
    auto minmax_tan = TangentMinMax(min_angle, max_angle);
    apply_scan_shadow_filter_functor func(thrust::raw_pointer_cast(ranges_.data()),
                                          minmax_tan.first, minmax_tan.second,
                                          GetAngleIncrement(),
                                          num_steps_,
                                          window, neighbors,
                                          remove_shadow_start_point,
                                          thrust::raw_pointer_cast(out->ranges_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(ranges_.size()), func);
    return out;
}

}
}