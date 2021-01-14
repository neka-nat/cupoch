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
#include "cupoch/geometry/map2d.h"
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace geometry {

Map2D::Map2D() : GeometryBase2D(Geometry::GeometryType::Map2D) {}

Map2D &Map2D::Clear() {
    map_.Clear();
    return *this;
}

bool Map2D::IsEmpty() const {
    return !map_.HasData();
}

Eigen::Vector2f Map2D::GetMinBound() const {
    return Eigen::Vector2f(0.0, 0.0);
}

Eigen::Vector2f Map2D::GetMaxBound() const {
    return Eigen::Vector2f(map_.width_ + map_.width_, map_.height_);
}

Eigen::Vector2f Map2D::GetCenter() const {
    return Eigen::Vector2f(map_.width_, map_.height_) * 0.5 + origin_;
}

AxisAlignedBoundingBox<2> Map2D::GetAxisAlignedBoundingBox() const {
    utility::LogError("Map2D::GetAxisAlignedBoundingBox is not supported");
    return AxisAlignedBoundingBox<2>();
}

Map2D &Map2D::Transform(const Eigen::Matrix3f &transformation) {
    utility::LogError("Map2D::Transform is not supported");
    return *this;
}

Map2D &Map2D::Translate(const Eigen::Vector2f &translation,
                                bool relative) {
    origin_ += translation;
    return *this;
}

Map2D &Map2D::Scale(const float scale, bool center) {
    cell_size_ *= scale;
    return *this;
}

Map2D &Map2D::Rotate(const Eigen::Matrix2f &R, bool center) {
    utility::LogError("Map2D::Rotate is not supported");
    return *this;
}

}
}