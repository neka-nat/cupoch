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
    __host__ __device__ DistanceVoxel(const Eigen::Vector3ui16& nearest_index, unsigned char state)
    : nearest_index_(nearest_index), state_(state) {}
    __host__ __device__ ~DistanceVoxel() {}

    __host__ __device__ bool IsNotSite() const { return state_ & NotSite ; };
    __host__ __device__ bool CheckHasNext() const { return state_ & HasNext ; };

public:
    Eigen::Vector3ui16 nearest_index_ = Eigen::Vector3ui16::Zero();
    unsigned char state_ = NotSite;
    float distance_ = 0;
};

class DistanceTransform : public DenseGrid<DistanceVoxel> {
public:
    DistanceTransform();
    DistanceTransform(float voxel_size, int resolution,
                      const Eigen::Vector3f &origin = Eigen::Vector3f::Zero());
    ~DistanceTransform();

    DistanceTransform &Reconstruct(float voxel_size, int resolution);

    DistanceTransform &ComputeEDT(const utility::device_vector<Eigen::Vector3i>& points);
    DistanceTransform &ComputeEDT(const VoxelGrid& voxelgrid);
    DistanceTransform &ComputeVoronoiDiagram(const utility::device_vector<Eigen::Vector3i>& points);
    DistanceTransform &ComputeVoronoiDiagram(const VoxelGrid& voxelgrid);

private:
    utility::device_vector<DistanceVoxel> buffer_;
};

}
}