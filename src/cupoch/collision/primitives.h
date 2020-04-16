#pragma once
#include <memory>
#include <Eigen/Core>

namespace cupoch {

namespace geometry {
class TriangleMesh;
class VoxelGrid;
}

namespace collision {

class Primitive {
public:
    enum class PrimitiveType {
        Unspecified = 0,
        Box = 1,
        Sphere = 2,
        Cylinder = 3,
        Cone = 4,
        Mesh = 5,
    };
    Primitive(PrimitiveType type);
    virtual ~Primitive();

    virtual std::shared_ptr<geometry::VoxelGrid> CreateVoxelGrid(float voxel_size) const = 0;
    virtual std::shared_ptr<geometry::VoxelGrid> CreateVoxelGridWithSweeping(
        float voxel_size, const Eigen::Matrix4f& dst, int sampling = 100) const = 0;
    virtual std::shared_ptr<geometry::TriangleMesh> CreateTriangleMesh() const = 0;
    PrimitiveType type_ = PrimitiveType::Unspecified;
    Eigen::Matrix4f transform_;
};

class Sphere : public Primitive {
public:
    Sphere();
    Sphere(float radius);
    virtual ~Sphere();

    std::shared_ptr<geometry::VoxelGrid> CreateVoxelGrid(float voxel_size) const override;
    std::shared_ptr<geometry::VoxelGrid> CreateVoxelGridWithSweeping(
        float voxel_size, const Eigen::Matrix4f& dst, int sampling = 100) const override;
    std::shared_ptr<geometry::TriangleMesh> CreateTriangleMesh() const override;
    float radius_;
};

}
}