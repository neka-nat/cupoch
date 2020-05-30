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

}
}