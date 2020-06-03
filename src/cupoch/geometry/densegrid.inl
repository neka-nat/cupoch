#include "cupoch/geometry/densegrid.h"
#include "cupoch/geometry/boundingvolume.h"

namespace cupoch {
namespace geometry {

template<class VoxelType>
DenseGrid<VoxelType>::DenseGrid(Geometry::GeometryType type) : Geometry3D(type) {}
template<class VoxelType>
DenseGrid<VoxelType>::DenseGrid(Geometry::GeometryType type, float voxel_size, int resolution, const Eigen::Vector3f& origin)
: Geometry3D(type), voxel_size_(voxel_size), resolution_(resolution), origin_(origin) {
    voxels_.resize(resolution_ * resolution_ * resolution_);
}
template<class VoxelType>
DenseGrid<VoxelType>::DenseGrid(Geometry::GeometryType type, const DenseGrid &src_grid)
: Geometry3D(type), voxel_size_(src_grid.voxel_size_),
 resolution_(src_grid.resolution_),
 origin_(src_grid.origin_),
 voxels_(src_grid.voxels_) {}
template<class VoxelType>
DenseGrid<VoxelType>::~DenseGrid() {}

template<class VoxelType>
DenseGrid<VoxelType> &DenseGrid<VoxelType>::Clear() {
    voxel_size_ = 0.0;
    resolution_ = 0;
    origin_ = Eigen::Vector3f::Zero();
    voxels_.clear();
    return *this;
}

template<class VoxelType>
bool DenseGrid<VoxelType>::IsEmpty() const { return voxels_.empty(); }

template<class VoxelType>
Eigen::Vector3f DenseGrid<VoxelType>::GetMinBound() const {
    float len = voxel_size_ * resolution_ * 0.5;
    return origin_ - Eigen::Vector3f::Constant(len);
}

template<class VoxelType>
Eigen::Vector3f DenseGrid<VoxelType>::GetMaxBound() const {
    float len = voxel_size_ * resolution_ * 0.5;
    return origin_ + Eigen::Vector3f::Constant(len);
}

template<class VoxelType>
Eigen::Vector3f DenseGrid<VoxelType>::GetCenter() const {
    return origin_;
}

template<class VoxelType>
AxisAlignedBoundingBox DenseGrid<VoxelType>::GetAxisAlignedBoundingBox() const {
    AxisAlignedBoundingBox box;
    box.min_bound_ = GetMinBound();
    box.max_bound_ = GetMaxBound();
    return box;
}

template<class VoxelType>
OrientedBoundingBox DenseGrid<VoxelType>::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
            GetAxisAlignedBoundingBox());
}

template<class VoxelType>
DenseGrid<VoxelType> &DenseGrid<VoxelType>::Transform(const Eigen::Matrix4f &transformation) {
    utility::LogError("DenseGrid::Transform is not supported");
    return *this;
}

template<class VoxelType>
DenseGrid<VoxelType> &DenseGrid<VoxelType>::Translate(const Eigen::Vector3f &translation,
                                               bool relative) {
    origin_ += translation;
    return *this;
}

template<class VoxelType>
DenseGrid<VoxelType> &DenseGrid<VoxelType>::Scale(const float scale, bool center) {
    voxel_size_ *= scale;
    return *this;
}

template<class VoxelType>
DenseGrid<VoxelType> &DenseGrid<VoxelType>::Rotate(const Eigen::Matrix3f &R, bool center) {
    utility::LogError("DenseGrid::Rotate is not supported");
    return *this;
}

template<class VoxelType>
DenseGrid<VoxelType> &DenseGrid<VoxelType>::Reconstruct(float voxel_size, int resolution) {
    voxel_size_ = voxel_size;
    resolution_ = resolution;
    voxels_.resize(resolution_ * resolution_ * resolution_, VoxelType());
    return *this;
}

template<class VoxelType>
int DenseGrid<VoxelType>::GetVoxelIndex(const Eigen::Vector3f& point) const {
    Eigen::Vector3f voxel_f = (point - origin_) / voxel_size_;
    int h_res = resolution_ / 2;
    Eigen::Vector3i voxel_idx = (Eigen::floor(voxel_f.array())).matrix().cast<int>() + Eigen::Vector3i::Constant(h_res);
    int idx = IndexOf(voxel_idx, resolution_);
    if (idx < 0 || idx >= resolution_ * resolution_ * resolution_) return -1;
    return idx;
}

template<class VoxelType>
thrust::tuple<bool, VoxelType> DenseGrid<VoxelType>::GetVoxel(const Eigen::Vector3f &point) const {
    auto idx = GetVoxelIndex(point);
    if (idx < 0) return thrust::make_tuple(false, VoxelType());
    VoxelType voxel = voxels_[idx];
    return thrust::make_tuple(true, voxel);
}

}
}