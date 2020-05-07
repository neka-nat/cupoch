#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/geometry_functor.h"

namespace cupoch {
namespace geometry {

template<class VoxelType>
VoxelGridBase<VoxelType>::VoxelGridBase(Geometry::GeometryType type) : Geometry3D(type) {}
template<class VoxelType>
VoxelGridBase<VoxelType>::VoxelGridBase(Geometry::GeometryType type, float voxel_size, const Eigen::Vector3f& origin)
 : Geometry3D(type), voxel_size_(voxel_size), origin_(origin_) {}
template<class VoxelType>
VoxelGridBase<VoxelType>::VoxelGridBase(Geometry::GeometryType type, const VoxelGridBase &src_voxel_grid)
 : Geometry3D(type), voxel_size_(src_voxel_grid.voxel_size_),
  origin_(src_voxel_grid.origin_),
  voxels_keys_(src_voxel_grid.voxels_keys_),
  voxels_values_(src_voxel_grid.voxels_values_) {}
template<class VoxelType>
VoxelGridBase<VoxelType>::~VoxelGridBase() {}

template<class VoxelType>
VoxelGridBase<VoxelType> &VoxelGridBase<VoxelType>::Clear() {
    voxel_size_ = 0.0;
    origin_ = Eigen::Vector3f::Zero();
    voxels_keys_.clear();
    voxels_values_.clear();
    return *this;
}

template<class VoxelType>
bool VoxelGridBase<VoxelType>::IsEmpty() const { return voxels_keys_.empty(); }

template<class VoxelType>
Eigen::Vector3f VoxelGridBase<VoxelType>::GetMinBound() const {
    if (voxels_keys_.empty()) {
        return origin_;
    } else {
        Eigen::Vector3i init = voxels_keys_[0];
        Eigen::Vector3i min_grid_index = thrust::reduce(voxels_keys_.begin(),
                voxels_keys_.end(), init, thrust::elementwise_minimum<Eigen::Vector3i>());
        return min_grid_index.cast<float>() * voxel_size_ + origin_;
    }
}

template<class VoxelType>
Eigen::Vector3f VoxelGridBase<VoxelType>::GetMaxBound() const {
    if (voxels_keys_.empty()) {
        return origin_;
    } else {
        Eigen::Vector3i init = voxels_keys_[0];
        Eigen::Vector3i max_grid_index = thrust::reduce(voxels_keys_.begin(),
                voxels_keys_.end(), init, thrust::elementwise_maximum<Eigen::Vector3i>());
        return (max_grid_index.cast<float>() + Eigen::Vector3f::Ones()) *
                       voxel_size_ +
               origin_;
    }
}

template<class VoxelType>
Eigen::Vector3f VoxelGridBase<VoxelType>::GetCenter() const {
    Eigen::Vector3f init(0, 0, 0);
    if (voxels_keys_.empty()) {
        return init;
    }
    compute_grid_center_functor func(voxel_size_, origin_);
    Eigen::Vector3f center = thrust::transform_reduce(voxels_keys_.begin(),
            voxels_keys_.end(), func, init, thrust::plus<Eigen::Vector3f>());
    center /= float(voxels_values_.size());
    return center;
}

template<class VoxelType>
AxisAlignedBoundingBox VoxelGridBase<VoxelType>::GetAxisAlignedBoundingBox() const {
    AxisAlignedBoundingBox box;
    box.min_bound_ = GetMinBound();
    box.max_bound_ = GetMaxBound();
    return box;
}

template<class VoxelType>
OrientedBoundingBox VoxelGridBase<VoxelType>::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromAxisAlignedBoundingBox(
            GetAxisAlignedBoundingBox());
}

template<class VoxelType>
VoxelGridBase<VoxelType> &VoxelGridBase<VoxelType>::Transform(const Eigen::Matrix4f &transformation) {
    utility::LogError("VoxelGridBase::Transform is not supported");
    return *this;
}

template<class VoxelType>
VoxelGridBase<VoxelType> &VoxelGridBase<VoxelType>::Translate(const Eigen::Vector3f &translation,
                                               bool relative) {
    origin_ += translation;
    return *this;
}

template<class VoxelType>
VoxelGridBase<VoxelType> &VoxelGridBase<VoxelType>::Scale(const float scale, bool center) {
    voxel_size_ *= scale;
    return *this;
}

template<class VoxelType>
VoxelGridBase<VoxelType> &VoxelGridBase<VoxelType>::Rotate(const Eigen::Matrix3f &R, bool center) {
    utility::LogError("VoxelGridBase::Rotate is not supported");
    return *this;
}

}
}