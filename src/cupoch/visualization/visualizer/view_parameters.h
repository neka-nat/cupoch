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