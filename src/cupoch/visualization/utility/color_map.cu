#include "cupoch/visualization/utility/color_map.h"

#include "cupoch/utility/console.h"
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

namespace cupoch {

namespace {
using namespace visualization;

class GlobalColorMapSingleton {
private:
    GlobalColorMapSingleton() : color_map_(thrust::device_malloc<ColorMapJet>(1)) {
        utility::LogDebug("Global colormap init.");
    }
    GlobalColorMapSingleton(const GlobalColorMapSingleton &) = delete;
    GlobalColorMapSingleton &operator=(const GlobalColorMapSingleton &) =
            delete;

public:
    ~GlobalColorMapSingleton() {
        thrust::device_free(color_map_);
        utility::LogDebug("Global colormap destruct.");
    }

public:
    static GlobalColorMapSingleton &GetInstance() {
        static GlobalColorMapSingleton singleton;
        return singleton;
    }

public:
    thrust::device_ptr<ColorMap> color_map_;
};

}  // unnamed namespace

namespace visualization {

__host__ __device__
Eigen::Vector3f ColorMapGray::GetColor(float value) const {
    return Eigen::Vector3f(value, value, value);
}

__host__ __device__
Eigen::Vector3f ColorMapJet::GetColor(float value) const {
    return Eigen::Vector3f(JetBase(value * 2.0 - 1.5),   // red
                           JetBase(value * 2.0 - 1.0),   // green
                           JetBase(value * 2.0 - 0.5));  // blue
}

__host__ __device__
Eigen::Vector3f ColorMapSummer::GetColor(float value) const {
    return Eigen::Vector3f(Interpolate(value, 0.0, 0.0, 1.0, 1.0),
                           Interpolate(value, 0.5, 0.0, 1.0, 1.0), 0.4);
}

__host__ __device__
Eigen::Vector3f ColorMapWinter::GetColor(float value) const {
    return Eigen::Vector3f(0.0, Interpolate(value, 0.0, 0.0, 1.0, 1.0),
                           Interpolate(value, 1.0, 0.0, 0.5, 1.0));
}

__host__ __device__
Eigen::Vector3f ColorMapHot::GetColor(float value) const {
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

const thrust::device_ptr<const ColorMap> GetGlobalColorMap() {
    return GlobalColorMapSingleton::GetInstance().color_map_;
}

void SetGlobalColorMap(ColorMap::ColorMapOption option) {
    thrust::device_free(GlobalColorMapSingleton::GetInstance().color_map_);
    switch (option) {
        case ColorMap::ColorMapOption::Gray:
            GlobalColorMapSingleton::GetInstance().color_map_ = thrust::device_malloc<ColorMapGray>(1);
            break;
        case ColorMap::ColorMapOption::Summer:
            GlobalColorMapSingleton::GetInstance().color_map_ = thrust::device_malloc<ColorMapSummer>(1);
            break;
        case ColorMap::ColorMapOption::Winter:
            GlobalColorMapSingleton::GetInstance().color_map_ = thrust::device_malloc<ColorMapWinter>(1);
            break;
        case ColorMap::ColorMapOption::Hot:
            GlobalColorMapSingleton::GetInstance().color_map_ = thrust::device_malloc<ColorMapHot>(1);
            break;
        case ColorMap::ColorMapOption::Jet:
        default:
            GlobalColorMapSingleton::GetInstance().color_map_ = thrust::device_malloc<ColorMapJet>(1);
            break;
    }
}

}  // namespace visualization
}  // namespace cupoch