#pragma once
#include <memory>
#include <Eigen/Core>
#include <cuda_runtime.h>

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
    __host__ __device__ Primitive()
    : type_(PrimitiveType::Unspecified), transform_(Eigen::Matrix4f::Identity()) {};
    __host__ __device__ Primitive(PrimitiveType type)
    : type_(type), transform_(Eigen::Matrix4f::Identity()) {};
    __host__ __device__ ~Primitive() {};

    PrimitiveType type_ = PrimitiveType::Unspecified;
    Eigen::Matrix4f transform_;
};

class Sphere : public Primitive {
public:
    __host__ __device__ Sphere() : Primitive(Primitive::PrimitiveType::Sphere), radius_(0.0) {};
    __host__ __device__ Sphere(float radius) : Primitive(Primitive::PrimitiveType::Sphere), radius_(radius) {};
    __host__ __device__ Sphere(float radius, const Eigen::Vector3f& center)
    : Primitive(Primitive::PrimitiveType::Sphere), radius_(radius) {
        transform_.block<3, 1>(0, 3) = center;
    };
    __host__ __device__ ~Sphere() {};
    float radius_;
};

std::shared_ptr<geometry::VoxelGrid> CreateVoxelGrid(const Primitive& primitive, float voxel_size);
std::shared_ptr<geometry::VoxelGrid> CreateVoxelGridWithSweeping(const Primitive& primitive, 
    float voxel_size, const Eigen::Matrix4f& dst, int sampling = 100);
std::shared_ptr<geometry::TriangleMesh> CreateTriangleMesh(const Primitive& primitive);

}
}