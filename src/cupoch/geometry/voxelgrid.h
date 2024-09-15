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

#include <thrust/transform_reduce.h>

#include <Eigen/Core>
#include <memory>

#include "cupoch/geometry/geometry_base.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/helper.h"

namespace cupoch {

namespace camera {
class PinholeCameraParameters;
}

namespace geometry {

class PointCloud;
class TriangleMesh;
class OrientedBoundingBox;
class Image;
class OccupancyGrid;

__device__ const int INVALID_VOXEL_INDEX = std::numeric_limits<int>::min();

class Voxel {
public:
    __host__ __device__ Voxel() {}
    __host__ __device__ Voxel(const Eigen::Vector3i &grid_index)
        : grid_index_(grid_index) {}
    __host__ __device__ Voxel(const Eigen::Vector3f &color) : color_(color) {}
    __host__ __device__ Voxel(const Eigen::Vector3i &grid_index,
                              const Eigen::Vector3f &color)
        : grid_index_(grid_index), color_(color) {}
    __host__ __device__ ~Voxel() {}

public:
    Eigen::Vector3i grid_index_ = Eigen::Vector3i(0, 0, 0);
    Eigen::Vector3f color_ = Eigen::Vector3f(1.0, 1.0, 1.0);
};

struct add_voxel_color_functor {
    __device__ thrust::tuple<Voxel, int> operator()(
            const thrust::tuple<Voxel, int> &x,
            const thrust::tuple<Voxel, int> &y) const {
        Voxel ans;
        ans.grid_index_ = thrust::get<0>(x).grid_index_;
        ans.color_ = thrust::get<0>(x).color_ + thrust::get<0>(y).color_;
        return thrust::make_tuple(ans, thrust::get<1>(x) + thrust::get<1>(y));
    }
};

struct divide_voxel_color_functor {
    __device__ Voxel operator()(const Voxel &x, int y) const {
        Voxel ans;
        ans.grid_index_ = x.grid_index_;
        ans.color_ = x.color_ / y;
        return ans;
    }
};

class VoxelGrid : public GeometryBase3D {
public:
    VoxelGrid();
    VoxelGrid(const VoxelGrid &src_voxel_grid);
    ~VoxelGrid();

    std::pair<thrust::host_vector<Eigen::Vector3i>, thrust::host_vector<Voxel>>
    GetVoxels() const;
    void SetVoxels(const thrust::host_vector<Eigen::Vector3i> &voxels_keys,
                   const thrust::host_vector<Voxel> &voxels_values);
    void SetVoxels(const std::vector<Eigen::Vector3i> &voxels_keys,
                   const std::vector<Voxel> &voxels_values);

    VoxelGrid &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3f GetMinBound() const override;
    Eigen::Vector3f GetMaxBound() const override;
    Eigen::Vector3f GetCenter() const override;
    AxisAlignedBoundingBox<3> GetAxisAlignedBoundingBox() const override;
    OrientedBoundingBox GetOrientedBoundingBox() const;
    VoxelGrid &Transform(const Eigen::Matrix4f &transformation) override;
    VoxelGrid &Translate(const Eigen::Vector3f &translation,
                         bool relative = true) override;
    VoxelGrid &Scale(const float scale, bool center = true) override;
    VoxelGrid &Rotate(const Eigen::Matrix3f &R, bool center = true) override;

    VoxelGrid &operator+=(const VoxelGrid &voxelgrid);
    VoxelGrid operator+(const VoxelGrid &voxelgrid) const;

    bool HasVoxels() const { return voxels_keys_.size() > 0; }
    bool HasColors() const {
        return true;  // By default, the colors are (1.0, 1.0, 1.0)
    }
    Eigen::Vector3i GetVoxel(const Eigen::Vector3f &point) const;

    // Function that returns the 3d coordinates of the queried voxel center
    Eigen::Vector3f GetVoxelCenterCoordinate(const Eigen::Vector3i &idx) const;

    /// Add a voxel with specified grid index and color
    void AddVoxel(const Voxel &voxel);
    void AddVoxels(const utility::device_vector<Voxel> &voxels);
    void AddVoxels(const thrust::host_vector<Voxel> &voxels);

    /// Assigns each voxel in the VoxelGrid the same color \param color.
    VoxelGrid &PaintUniformColor(const Eigen::Vector3f &color);

    VoxelGrid &PaintIndexedColor(const utility::device_vector<size_t> &indices,
                                 const Eigen::Vector3f &color);

    /// Return a vector of 3D coordinates that define the indexed voxel cube.
    std::array<Eigen::Vector3f, 8> GetVoxelBoundingPoints(
            const Eigen::Vector3i &index) const;

    // Element-wise check if a query in the list is included in the VoxelGrid
    // Queries are double precision and are mapped to the closest voxel.
    std::vector<bool> CheckIfIncluded(
            const std::vector<Eigen::Vector3f> &queries);

    /// Remove all voxels from the VoxelGrid where none of the boundary points
    /// of the voxel projects to depth value that is smaller, or equal than the
    /// projected depth of the boundary point. If keep_voxels_outside_image is
    /// true then voxels are only carved if all boundary points project to a
    /// valid image location.
    VoxelGrid &CarveDepthMap(
            const Image &depth_map,
            const camera::PinholeCameraParameters &camera_parameter,
            bool keep_voxels_outside_image);

    /// Remove all voxels from the VoxelGrid where none of the boundary points
    /// of the voxel projects to a valid mask pixel (pixel value > 0). If
    /// keep_voxels_outside_image is true then voxels are only carved if
    /// all boundary points project to a valid image location.
    VoxelGrid &CarveSilhouette(
            const Image &silhouette_mask,
            const camera::PinholeCameraParameters &camera_parameter,
            bool keep_voxels_outside_image);
    
    /// Selects all voxels from src by given \param indices and copies them to temp grid.
    /// if \param invert is set to true, deselects all voxels given by \param indices, copies remaing voxels
    /// from this to temp grid and returns it.
    std::shared_ptr<VoxelGrid> SelectByIndex(
                                              const utility::device_vector<size_t> &indices, 
                                              bool invert);        

    // Creates a voxel grid where every voxel is set (hence dense). This is a
    // useful starting point for voxel carving.
    static std::shared_ptr<VoxelGrid> CreateDense(const Eigen::Vector3f &origin,
                                                  float voxel_size,
                                                  float width,
                                                  float height,
                                                  float depth);

    // Creates a VoxelGrid from a given PointCloud. The color value of a given
    // voxel is the average color value of the points that fall into it (if the
    // PointCloud has colors).
    // The bounds of the created VoxelGrid are computed from the PointCloud.
    static std::shared_ptr<VoxelGrid> CreateFromPointCloud(
            const PointCloud &input, float voxel_size);

    // Creates a VoxelGrid from a given PointCloud. The color value of a given
    // voxel is the average color value of the points that fall into it (if the
    // PointCloud has colors).
    // The bounds of the created VoxelGrid are defined by the given parameters.
    static std::shared_ptr<VoxelGrid> CreateFromPointCloudWithinBounds(
            const PointCloud &input,
            float voxel_size,
            const Eigen::Vector3f &min_bound,
            const Eigen::Vector3f &max_bound);

    // Creates a VoxelGrid from a given TriangleMesh. No color information is
    // converted. The bounds of the created VoxelGrid are computed from the
    // TriangleMesh..
    static std::shared_ptr<VoxelGrid> CreateFromTriangleMesh(
            const TriangleMesh &input, float voxel_size);

    // Creates a VoxelGrid from a given TriangleMesh. No color information is
    // converted. The bounds of the created VoxelGrid are defined by the given
    // parameters..
    static std::shared_ptr<VoxelGrid> CreateFromTriangleMeshWithinBounds(
            const TriangleMesh &input,
            float voxel_size,
            const Eigen::Vector3f &min_bound,
            const Eigen::Vector3f &max_bound);

    static std::shared_ptr<VoxelGrid> CreateFromOccupancyGrid(
            const OccupancyGrid &input);

public:
    float voxel_size_ = 0.0;
    Eigen::Vector3f origin_ = Eigen::Vector3f::Zero();
    utility::device_vector<Eigen::Vector3i> voxels_keys_;
    utility::device_vector<Voxel> voxels_values_;
};

}  // namespace geometry
}  // namespace cupoch