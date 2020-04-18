#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/camera/pinhole_camera_intrinsic.h"

namespace cupoch {

namespace geometry {

class OccupancyVoxel : public Voxel {
public:
    __host__ __device__ OccupancyVoxel() : Voxel() {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i &grid_index)
        : Voxel(grid_index) {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i &grid_index, float prob_log)
        : Voxel(grid_index), prob_log_(prob_log) {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i &grid_index,
                                       const Eigen::Vector3f &color,
                                       float prob_log = 0.0)
        : Voxel(grid_index, color), prob_log_(prob_log) {}
    __host__ __device__ ~OccupancyVoxel() {}

public:
    float prob_log_ = 0;
};

class OccupancyGrid : Geometry3D {
public:
    OccupancyGrid();
    ~OccupancyGrid();
    OccupancyGrid(const OccupancyGrid& other);
    void Insert(const utility::device_vector<Eigen::Vector3f>& points, const Eigen::Vector3f& viewpoint);
    void Insert(const PointCloud& pointcloud, const Eigen::Vector3f& viewpoint);
    void Insert(const Image& depth, const camera::PinholeCameraIntrinsic &intrinsic,
                const Eigen::Matrix4f &extrinsic);
private:
    void AddVoxels(const utility::device_vector<Eigen::Vector3i>& voxels, bool occupied = false);
public:
    float voxel_size_ = 0.0;
    Eigen::Vector3f origin_ = Eigen::Vector3f::Zero();
    utility::device_vector<Eigen::Vector3i> voxels_keys_;
    utility::device_vector<OccupancyVoxel> voxels_values_;
    float clamping_thres_min_ = -2.0;
    float clamping_thres_max_ = 3.5;
    float prob_hit_log_ = 0.85;
    float prob_miss_log_ = -0.4;
    float occ_prob_thres_log_ = 0.0;
};

}

}