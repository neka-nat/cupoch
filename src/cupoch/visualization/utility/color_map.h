#pragma once

#include <thrust/device_ptr.h>

#include <Eigen/Core>

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
    __host__ __device__ ColorMap() {}
    __host__ __device__ virtual ~ColorMap() {}

public:
    /// Function to get a color from a value in [0..1]
    __host__ __device__ virtual Eigen::Vector3f GetColor(float value) const = 0;

protected:
    __host__ __device__ float Interpolate(
            float value, float y0, float x0, float y1, float x1) const {
        if (value < x0) return y0;
        if (value > x1) return y1;
        return (value - x0) * (y1 - y0) / (x1 - x0) + y0;
    }
    __host__ __device__ Eigen::Vector3f Interpolate(float value,
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
    __host__ __device__ Eigen::Vector3f GetColor(float value) const final {
        return Eigen::Vector3f(value, value, value);
    }
};

/// See Matlab's Jet colormap
class ColorMapJet final : public ColorMap {
public:
    __host__ __device__ Eigen::Vector3f GetColor(float value) const final {
        return Eigen::Vector3f(JetBase(value * 2.0 - 1.5),   // red
                               JetBase(value * 2.0 - 1.0),   // green
                               JetBase(value * 2.0 - 0.5));  // blue
    }

protected:
    __host__ __device__ float JetBase(float value) const {
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
    __host__ __device__ Eigen::Vector3f GetColor(float value) const final {
        return Eigen::Vector3f(Interpolate(value, 0.0, 0.0, 1.0, 1.0),
                               Interpolate(value, 0.5, 0.0, 1.0, 1.0), 0.4);
    }
};

/// See Matlab's Winter colormap
class ColorMapWinter final : public ColorMap {
public:
    __host__ __device__ Eigen::Vector3f GetColor(float value) const final {
        return Eigen::Vector3f(0.0, Interpolate(value, 0.0, 0.0, 1.0, 1.0),
                               Interpolate(value, 1.0, 0.0, 0.5, 1.0));
    }
};

class ColorMapHot final : public ColorMap {
public:
    __host__ __device__ Eigen::Vector3f GetColor(float value) const final {
        Eigen::Vector3f edges[4] = {
                Eigen::Vector3f(1.0, 1.0, 1.0),
                Eigen::Vector3f(1.0, 1.0, 0.0),
                Eigen::Vector3f(1.0, 0.0, 0.0),
                Eigen::Vector3f(0.0, 0.0, 0.0),
        };
        if (value < 0.0) {
            return edges[0];
        } else if (value < 1.0 / 3.0) {
            return Interpolate(value, edges[0], 0.0, edges[1], 1.0 / 3.0);
        } else if (value < 2.0 / 3.0) {
            return Interpolate(value, edges[1], 1.0 / 3.0, edges[2], 2.0 / 3.0);
        } else if (value < 1.0) {
            return Interpolate(value, edges[2], 2.0 / 3.0, edges[3], 1.0);
        } else {
            return edges[3];
        }
    }
};

/// Interface functions
__host__ __device__ inline Eigen::Vector3f GetColorMapColor(
        float value, ColorMap::ColorMapOption option) {
    switch (option) {
        case ColorMap::ColorMapOption::Gray:
            return ColorMapGray().GetColor(value);
        case ColorMap::ColorMapOption::Summer:
            return ColorMapSummer().GetColor(value);
        case ColorMap::ColorMapOption::Winter:
            return ColorMapWinter().GetColor(value);
        case ColorMap::ColorMapOption::Hot:
            return ColorMapHot().GetColor(value);
        case ColorMap::ColorMapOption::Jet:
        default:
            return ColorMapJet().GetColor(value);
    }
}

ColorMap::ColorMapOption GetGlobalColorMapOption();
void SetGlobalColorMapOption(ColorMap::ColorMapOption option);

}  // namespace visualization
}  // namespace cupoch