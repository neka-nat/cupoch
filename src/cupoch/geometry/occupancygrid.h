#include "cupoch/geometry/voxelgrid.h"

namespace cupoch {

namespace geometry {

class OccupancyVoxel : public Voxel {
public:
    __host__ __device__ OccupancyVoxel() : Voxel(Eigen::Vector3f(0.0, 0.0, 1.0)) {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i &grid_index)
        : Voxel(grid_index, Eigen::Vector3f(0.0, 0.0, 1.0)) {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i &grid_index, float prob_log)
        : Voxel(grid_index, Eigen::Vector3f(0.0, 0.0, 1.0)), prob_log_(prob_log) {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i &grid_index,
                                       float prob_log,
                                       const Eigen::Vector3f &color)
        : Voxel(grid_index, color), prob_log_(prob_log) {}
    __host__ __device__ ~OccupancyVoxel() {}

public:
    float prob_log_ = 0;
};

class OccupancyGrid : public Geometry3D {
public:
    OccupancyGrid();
    OccupancyGrid(float voxel_size, const Eigen::Vector3f& origin = Eigen::Vector3f::Zero());
    ~OccupancyGrid();
    OccupancyGrid(const OccupancyGrid& other);
    OccupancyGrid &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3f GetMinBound() const override;
    Eigen::Vector3f GetMaxBound() const override;
    Eigen::Vector3f GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    OrientedBoundingBox GetOrientedBoundingBox() const;
    OccupancyGrid &Transform(const Eigen::Matrix4f &transformation) override;
    OccupancyGrid &Translate(const Eigen::Vector3f &translation,
                             bool relative = true) override;
    OccupancyGrid &Scale(const float scale, bool center = true) override;
    OccupancyGrid &Rotate(const Eigen::Matrix3f &R, bool center = true) override;

    bool HasVoxels() const { return voxels_keys_.size() > 0; }
    bool HasColors() const {
        return true;  // By default, the colors are (1.0, 1.0, 1.0)
    }
    bool IsOccupied(const Eigen::Vector3f &point) const;
    bool IsUnknown(const Eigen::Vector3f &point) const;
    int GetVoxelIndex(const Eigen::Vector3f& point) const;
    thrust::tuple<bool, OccupancyVoxel> GetVoxel(const Eigen::Vector3f &point) const;

    OccupancyGrid& Insert(const utility::device_vector<Eigen::Vector3f>& points,
                          const Eigen::Vector3f& viewpoint, float max_range = -1.0);
    OccupancyGrid& Insert(const thrust::host_vector<Eigen::Vector3f>& points,
                          const Eigen::Vector3f& viewpoint, float max_range = -1.0);
    OccupancyGrid& Insert(const PointCloud& pointcloud, const Eigen::Vector3f& viewpoint,
                          float max_range = -1.0);

    OccupancyGrid& AddVoxel(const Eigen::Vector3i& voxels, bool occupied = false);
    OccupancyGrid& AddVoxels(const utility::device_vector<Eigen::Vector3i>& voxels, bool occupied = false);
public:
    float voxel_size_ = 0.05;
    Eigen::Vector3f origin_ = Eigen::Vector3f::Zero();
    utility::device_vector<Eigen::Vector3i> voxels_keys_;
    utility::device_vector<OccupancyVoxel> voxels_values_;
    float clamping_thres_min_ = -2.0;
    float clamping_thres_max_ = 3.5;
    float prob_hit_log_ = 0.85;
    float prob_miss_log_ = -0.4;
    float occ_prob_thres_log_ = 0.0;
    bool visualize_free_area_ = true;
};

}

}