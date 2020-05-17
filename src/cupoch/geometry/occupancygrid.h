#pragma once
#include "cupoch/geometry/densegrid.h"

namespace cupoch {

namespace geometry {
class PointCloud;

class OccupancyVoxel {
public:
    __host__ __device__ OccupancyVoxel() {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i &grid_index)
        : grid_index_(grid_index.cast<unsigned short>()) {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i &grid_index, float prob_log)
        : grid_index_(grid_index.cast<unsigned short>()), prob_log_(prob_log) {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i &grid_index,
                                       float prob_log,
                                       const Eigen::Vector3f &color)
        : grid_index_(grid_index.cast<unsigned short>()), color_(color), prob_log_(prob_log) {}
    __host__ __device__ ~OccupancyVoxel() {}

public:
    Eigen::Vector3ui16 grid_index_ = Eigen::Vector3ui16::Zero();
    Eigen::Vector3f color_ = Eigen::Vector3f(0.0, 0.0, 1.0);
    float prob_log_ = std::numeric_limits<float>::quiet_NaN();
};

class OccupancyGrid : public DenseGrid<OccupancyVoxel> {
public:
    OccupancyGrid();
    OccupancyGrid(float voxel_size, int resolution = 512, const Eigen::Vector3f& origin = Eigen::Vector3f::Zero());
    ~OccupancyGrid();
    OccupancyGrid(const OccupancyGrid& other);

    OccupancyGrid &Clear() override;

    Eigen::Vector3f GetMinBound() const override;
    Eigen::Vector3f GetMaxBound() const override;

    bool HasVoxels() const { return voxels_.size() > 0; }
    bool HasColors() const {
        return true;  // By default, the colors are (1.0, 1.0, 1.0)
    }
    bool IsOccupied(const Eigen::Vector3f &point) const;
    bool IsUnknown(const Eigen::Vector3f &point) const;
    int GetVoxelIndex(const Eigen::Vector3f& point) const;
    thrust::tuple<bool, OccupancyVoxel> GetVoxel(const Eigen::Vector3f &point) const;
    size_t CountKnownVoxels() const;
    size_t CountFreeVoxels() const;
    size_t CountOccupiedVoxels() const;
    utility::device_vector<OccupancyVoxel> ExtractKnownVoxels() const;
    utility::device_vector<OccupancyVoxel> ExtractFreeVoxels() const;
    utility::device_vector<OccupancyVoxel> ExtractOccupiedVoxels() const;
    utility::device_vector<Eigen::Vector3i> ExtractKnownVoxelIndices() const;
    utility::device_vector<Eigen::Vector3i> ExtractFreeVoxelIndices() const;
    utility::device_vector<Eigen::Vector3i> ExtractOccupiedVoxelIndices() const;
    void ExtractKnownVoxelIndices(utility::device_vector<Eigen::Vector3i>& indices) const;

    OccupancyGrid& ReconstructVoxels(float voxel_size, int resolution);

    OccupancyGrid& Insert(const utility::device_vector<Eigen::Vector3f>& points,
                          const Eigen::Vector3f& viewpoint, float max_range = -1.0);
    OccupancyGrid& Insert(const thrust::host_vector<Eigen::Vector3f>& points,
                          const Eigen::Vector3f& viewpoint, float max_range = -1.0);
    OccupancyGrid& Insert(const PointCloud& pointcloud, const Eigen::Vector3f& viewpoint,
                          float max_range = -1.0);

    OccupancyGrid& AddVoxel(const Eigen::Vector3i& voxels, bool occupied = false);
    OccupancyGrid& AddVoxels(const utility::device_vector<Eigen::Vector3i>& voxels, bool occupied = false);
public:
    Eigen::Vector3ui16 min_bound_ = Eigen::Vector3ui16::Zero();
    Eigen::Vector3ui16 max_bound_ = Eigen::Vector3ui16::Zero();
    float clamping_thres_min_ = -2.0;
    float clamping_thres_max_ = 3.5;
    float prob_hit_log_ = 0.85;
    float prob_miss_log_ = -0.4;
    float occ_prob_thres_log_ = 0.0;
    bool visualize_free_area_ = true;
};

}

}