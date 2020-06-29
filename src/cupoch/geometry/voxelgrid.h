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

__device__ const int INVALID_VOXEL_INDEX = std::numeric_limits<int>::min();

class Voxel {
public:
    __host__ __device__ Voxel() {}
    __host__ __device__ Voxel(const Eigen::Vector3i &grid_index)
        : grid_index_(grid_index) {}
    __host__ __device__ Voxel(const Eigen::Vector3f &color)
        : color_(color) {}
    __host__ __device__ Voxel(const Eigen::Vector3i &grid_index,
                              const Eigen::Vector3f &color)
        : grid_index_(grid_index), color_(color) {}
    __host__ __device__ ~Voxel() {}

public:
    Eigen::Vector3i grid_index_ = Eigen::Vector3i(0, 0, 0);
    Eigen::Vector3f color_ = Eigen::Vector3f(1.0, 1.0, 1.0);
};

class VoxelGrid : public GeometryBase<3> {
public:
    VoxelGrid();
    VoxelGrid(const VoxelGrid &src_voxel_grid);
    ~VoxelGrid();

    thrust::pair<thrust::host_vector<Eigen::Vector3i>, thrust::host_vector<Voxel>> GetVoxels() const;
    void SetVoxels(const thrust::host_vector<Eigen::Vector3i>& voxels_keys, const thrust::host_vector<Voxel>& voxels_values);

    VoxelGrid &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3f GetMinBound() const override;
    Eigen::Vector3f GetMaxBound() const override;
    Eigen::Vector3f GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
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

    /// Return a vector of 3D coordinates that define the indexed voxel cube.
    std::array<Eigen::Vector3f, 8> GetVoxelBoundingPoints(
            const Eigen::Vector3i &index) const;

    // Element-wise check if a query in the list is included in the VoxelGrid
    // Queries are double precision and are mapped to the closest voxel.
    thrust::host_vector<bool> CheckIfIncluded(
            const thrust::host_vector<Eigen::Vector3f> &queries);

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

public:
    float voxel_size_ = 0.0;
    Eigen::Vector3f origin_ = Eigen::Vector3f::Zero();
    utility::device_vector<Eigen::Vector3i> voxels_keys_;
    utility::device_vector<Voxel> voxels_values_;
};

}  // namespace geometry
}  // namespace cupoch