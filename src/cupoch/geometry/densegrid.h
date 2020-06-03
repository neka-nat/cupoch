#pragma once
#include "cupoch/geometry/geometry3d.h"

namespace cupoch {
namespace geometry {

template <class VoxelType>
class DenseGrid : public Geometry3D {
public:
    DenseGrid(Geometry::GeometryType type);
    DenseGrid(Geometry::GeometryType type, float voxel_size, int resolution, const Eigen::Vector3f& origin);
    DenseGrid(Geometry::GeometryType type, const DenseGrid &src_grid);
    virtual ~DenseGrid();

    virtual DenseGrid &Clear();
    virtual bool IsEmpty() const;
    virtual Eigen::Vector3f GetMinBound() const;
    virtual Eigen::Vector3f GetMaxBound() const;
    virtual Eigen::Vector3f GetCenter() const;
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const;
    virtual OrientedBoundingBox GetOrientedBoundingBox() const;
    virtual DenseGrid &Transform(const Eigen::Matrix4f &transformation);
    virtual DenseGrid &Translate(const Eigen::Vector3f &translation,
                                     bool relative = true);
    virtual DenseGrid &Scale(const float scale, bool center = true);
    virtual DenseGrid &Rotate(const Eigen::Matrix3f &R, bool center = true);

    virtual DenseGrid& Reconstruct(float voxel_size, int resolution);
public:
    float voxel_size_ = 0.0;
    int resolution_ = 0;
    Eigen::Vector3f origin_ = Eigen::Vector3f::Zero();
    utility::device_vector<VoxelType> voxels_;
};

}  // namespace geometry
}  // namespace cupoch