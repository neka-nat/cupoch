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
#include "cupoch/geometry/densegrid.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace geometry {
class VoxelGrid;

class DistanceVoxel {
public:
    enum State {
        HasNext = 1,
        NotSite = 1 << 1,
    };
    __host__ __device__ DistanceVoxel() {}
    __host__ __device__ DistanceVoxel(const Eigen::Vector3ui16 &nearest_index,
                                      unsigned char state)
        : nearest_index_(nearest_index), state_(state) {}
    __host__ __device__ ~DistanceVoxel() {}

    __host__ __device__ bool IsNotSite() const { return state_ & NotSite; };
    __host__ __device__ bool CheckHasNext() const { return state_ & HasNext; };

public:
    Eigen::Vector3ui16 nearest_index_ = Eigen::Vector3ui16::Zero();
    unsigned char state_ = NotSite;
    float distance_ = 0;
};

class DistanceTransform : public DenseGrid<DistanceVoxel> {
public:
    DistanceTransform();
    DistanceTransform(float voxel_size,
                      int resolution,
                      const Eigen::Vector3f &origin = Eigen::Vector3f::Zero());
    DistanceTransform(const DistanceTransform& other);
    ~DistanceTransform();

    DistanceTransform &Reconstruct(float voxel_size, int resolution);

    DistanceTransform &ComputeEDT(
            const utility::device_vector<Eigen::Vector3i> &points);
    DistanceTransform &ComputeEDT(const VoxelGrid &voxelgrid);
    DistanceTransform &ComputeVoronoiDiagram(
            const utility::device_vector<Eigen::Vector3i> &points);
    DistanceTransform &ComputeVoronoiDiagram(const VoxelGrid &voxelgrid);

    float GetDistance(const Eigen::Vector3f& query) const;
    utility::device_vector<float> GetDistances(const utility::device_vector<Eigen::Vector3f>& queries) const;

private:
    utility::device_vector<DistanceVoxel> buffer_;
};

}  // namespace geometry
}  // namespace cupoch