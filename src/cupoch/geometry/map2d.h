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
#include "cupoch/geometry/image.h"

namespace cupoch {
namespace geometry {

class Map2D : public GeometryBase2D {
public:
    Map2D();
    ~Map2D() { map_.Clear(); };
    Map2D(const Map2D& other);

    Map2D &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector2f GetMinBound() const override;
    Eigen::Vector2f GetMaxBound() const override;
    Eigen::Vector2f GetCenter() const override;
    AxisAlignedBoundingBox<2> GetAxisAlignedBoundingBox() const override;
    Map2D &Transform(const Eigen::Matrix3f &transformation) override;
    Map2D &Translate(const Eigen::Vector2f &translation,
                         bool relative = true) override;
    Map2D &Scale(const float scale, bool center = true) override;
    Map2D &Rotate(const Eigen::Matrix2f &R, bool center = true) override;

public:
    Image map_;
    float cell_size_;
    Eigen::Vector2f origin_ = Eigen::Vector2f::Zero();
};

}
} // namespace cupoch
