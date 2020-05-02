#pragma once

#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/integration/tsdfvolume.h"

namespace cupoch {

namespace geometry {

class TSDFVoxel : public Voxel {
public:
    __host__ __device__ TSDFVoxel() : Voxel() {}
    __host__ __device__ TSDFVoxel(const Eigen::Vector3i &grid_index)
        : Voxel(grid_index) {}
    __host__ __device__ TSDFVoxel(const Eigen::Vector3i &grid_index,
                                  const Eigen::Vector3f &color)
        : Voxel(grid_index, color) {}
    __host__ __device__ ~TSDFVoxel() {}

public:
    float tsdf_ = 0;
    float weight_ = 0;
};

}  // namespace geometry

namespace integration {

class UniformTSDFVolume : public TSDFVolume {
public:
    UniformTSDFVolume(float length,
                      int resolution,
                      float sdf_trunc,
                      TSDFVolumeColorType color_type,
                      const Eigen::Vector3f &origin = Eigen::Vector3f::Zero());
    ~UniformTSDFVolume() override;
    UniformTSDFVolume(const UniformTSDFVolume &other);

public:
    void Reset() override;
    void Integrate(const geometry::RGBDImage &image,
                   const camera::PinholeCameraIntrinsic &intrinsic,
                   const Eigen::Matrix4f &extrinsic) override;
    std::shared_ptr<geometry::PointCloud> ExtractPointCloud() override;
    std::shared_ptr<geometry::TriangleMesh> ExtractTriangleMesh() override;

    /// Debug function to extract the voxel data into a VoxelGrid
    std::shared_ptr<geometry::PointCloud> ExtractVoxelPointCloud() const;
    std::shared_ptr<geometry::VoxelGrid> ExtractVoxelGrid() const;

    /// Faster Integrate function that uses depth_to_camera_distance_multiplier
    /// precomputed from camera intrinsic
    void IntegrateWithDepthToCameraDistanceMultiplier(
            const geometry::RGBDImage &image,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4f &extrinsic,
            const geometry::Image &depth_to_camera_distance_multiplier);

public:
    utility::device_vector<geometry::TSDFVoxel> voxels_;
    Eigen::Vector3f origin_;
    float length_;
    int resolution_;
    int voxel_num_;
};

}  // namespace integration
}  // namespace cupoch