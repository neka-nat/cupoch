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
#include <thrust/tuple.h>

#include "cupoch/geometry/geometry_base.h"
#include "cupoch/utility/device_vector.h"

namespace cupoch {
namespace geometry {

class OrientedBoundingBox;

template <class VoxelType>
class DenseGrid : public GeometryBase3D {
public:
    DenseGrid(Geometry::GeometryType type);
    DenseGrid(Geometry::GeometryType type,
              float voxel_size,
              int resolution,
              const Eigen::Vector3f &origin);
    DenseGrid(Geometry::GeometryType type, const DenseGrid &src_grid);
    virtual ~DenseGrid();

    virtual DenseGrid &Clear();
    virtual bool IsEmpty() const;
    virtual Eigen::Vector3f GetMinBound() const;
    virtual Eigen::Vector3f GetMaxBound() const;
    virtual Eigen::Vector3f GetCenter() const;
    virtual AxisAlignedBoundingBox<3> GetAxisAlignedBoundingBox() const;
    virtual OrientedBoundingBox GetOrientedBoundingBox() const;
    virtual DenseGrid &Transform(const Eigen::Matrix4f &transformation);
    virtual DenseGrid &Translate(const Eigen::Vector3f &translation,
                                 bool relative = true);
    virtual DenseGrid &Scale(const float scale, bool center = true);
    virtual DenseGrid &Rotate(const Eigen::Matrix3f &R, bool center = true);

    virtual DenseGrid &Reconstruct(float voxel_size, int resolution);

    int GetVoxelIndex(const Eigen::Vector3f &point) const;
    thrust::tuple<bool, VoxelType> GetVoxel(const Eigen::Vector3f &point) const;

public:
    float voxel_size_ = 0.0;
    int resolution_ = 0;
    Eigen::Vector3f origin_ = Eigen::Vector3f::Zero();
    utility::device_vector<VoxelType> voxels_;
};

}  // namespace geometry
}  // namespace cupoch