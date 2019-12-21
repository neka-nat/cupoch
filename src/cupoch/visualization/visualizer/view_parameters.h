#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "cupoch/utility/ijson_convertible.h"

namespace cupoch {
namespace visualization {

class ViewParameters : public utility::IJsonConvertible {
public:
    typedef Eigen::Matrix<float, 17, 4, Eigen::RowMajor> Matrix17x4f;
    typedef Eigen::Matrix<float, 17, 1> Vector17f;
    typedef Eigen::aligned_allocator<Matrix17x4f> Matrix17x4f_allocator;

public:
    ViewParameters()
        : field_of_view_(0),
          zoom_(0),
          lookat_(0, 0, 0),
          up_(0, 0, 0),
          front_(0, 0, 0),
          boundingbox_min_(0, 0, 0),
          boundingbox_max_(0, 0, 0) {}
    ~ViewParameters() override {}

public:
    Vector17f ConvertToVector17f();
    void ConvertFromVector17f(const Vector17f &v);
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    float field_of_view_;
    float zoom_;
    Eigen::Vector3f lookat_;
    Eigen::Vector3f up_;
    Eigen::Vector3f front_;
    Eigen::Vector3f boundingbox_min_;
    Eigen::Vector3f boundingbox_max_;
};

}  // namespace visualization
}  // namespace cupoch