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

#if !defined(__CUDACC__)
#if !defined(__host__)
#define __host__
#endif
#if !defined(__device__)
#define __device__
#endif
#endif

namespace cupoch {
namespace geometry {

class Geometry {
public:
    enum class GeometryType {
        Unspecified = 0,
        /// PointCloud
        PointCloud = 1,
        /// VoxelGrid
        VoxelGrid = 2,
        /// OccupancyGrid
        OccupancyGrid = 3,
        /// DistanceTransform
        DistanceTransform = 4,
        /// LineSet
        LineSet = 5,
        /// Graph
        Graph = 6,
        /// MeshBase
        MeshBase = 7,
        /// TriangleMesh
        TriangleMesh = 8,
        /// Image
        Image = 9,
        /// RGBDImage
        RGBDImage = 10,
        /// Map2D
        Map2D = 11,
        /// OrientedBoundingBox
        OrientedBoundingBox = 12,
        /// AxisAlignedBoundingBox
        AxisAlignedBoundingBox = 13,
        /// LaserScanBuffer
        LaserScanBuffer = 14,
    };

public:
    __host__ __device__ ~Geometry() {}  // non-virtual

protected:
    __host__ __device__ Geometry(GeometryType type, int dimension)
        : geometry_type_(type), dimension_(dimension) {}

public:
    virtual Geometry &Clear() = 0;
    virtual bool IsEmpty() const = 0;
    GeometryType GetGeometryType() const { return geometry_type_; }
    int Dimension() const { return dimension_; }

private:
    GeometryType geometry_type_ = GeometryType::Unspecified;
    int dimension_ = 3;
};

}  // namespace geometry
}  // namespace cupoch