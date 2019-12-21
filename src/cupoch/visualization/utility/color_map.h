#pragma once

#include <Eigen/Core>
#include <thrust/device_ptr.h>

namespace cupoch {
namespace visualization {

class ColorMap {
public:
    enum class ColorMapOption {
        Gray = 0,
        Jet = 1,
        Summer = 2,
        Winter = 3,
        Hot = 4,
    };

public:
    ColorMap() {}
    virtual ~ColorMap() {}

public:
    /// Function to get a color from a value in [0..1]
    __host__ __device__
    virtual Eigen::Vector3f GetColor(float value) const = 0;

protected:
    __host__ __device__
    float Interpolate(
            float value, float y0, float x0, float y1, float x1) const {
        if (value < x0) return y0;
        if (value > x1) return y1;
        return (value - x0) * (y1 - y0) / (x1 - x0) + y0;
    }
    __host__ __device__
    Eigen::Vector3f Interpolate(float value,
                                const Eigen::Vector3f &y0,
                                float x0,
                                const Eigen::Vector3f &y1,
                                float x1) const {
        if (value < x0) return y0;
        if (value > x1) return y1;
        return (value - x0) * (y1 - y0) / (x1 - x0) + y0;
    }
};

class ColorMapGray final : public ColorMap {
public:
    __host__ __device__
    Eigen::Vector3f GetColor(float value) const final;
};

/// See Matlab's Jet colormap
class ColorMapJet final : public ColorMap {
public:
    __host__ __device__
    Eigen::Vector3f GetColor(float value) const final;

protected:
    __host__ __device__
    float JetBase(float value) const {
        if (value <= -0.75) {
            return 0.0;
        } else if (value <= -0.25) {
            return Interpolate(value, 0.0, -0.75, 1.0, -0.25);
        } else if (value <= 0.25) {
            return 1.0;
        } else if (value <= 0.75) {
            return Interpolate(value, 1.0, 0.25, 0.0, 0.75);
        } else {
            return 0.0;
        }
    }
};

/// See Matlab's Summer colormap
class ColorMapSummer final : public ColorMap {
public:
    __host__ __device__
    Eigen::Vector3f GetColor(float value) const final;
};

/// See Matlab's Winter colormap
class ColorMapWinter final : public ColorMap {
public:
    __host__ __device__
    Eigen::Vector3f GetColor(float value) const final;
};

class ColorMapHot final : public ColorMap {
public:
    __host__ __device__
    Eigen::Vector3f GetColor(float value) const final;
};

/// Interface functions
const thrust::device_ptr<const ColorMap> GetGlobalColorMap();
void SetGlobalColorMap(ColorMap::ColorMapOption option);

}  // namespace visualization
}  // namespace cupoch