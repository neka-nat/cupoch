#pragma once
#include <memory>
#include "cupoch/geometry/boundingvolume.h"

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
        Capsule = 3,
    };
    __host__ __device__ Primitive()
    : type_(PrimitiveType::Unspecified), transform_(Eigen::Matrix4f::Identity()) {};
    __host__ __device__ Primitive(PrimitiveType type)
    : type_(type), transform_(Eigen::Matrix4f::Identity()) {};
    __host__ __device__ Primitive(PrimitiveType type, const Eigen::Matrix4f& transform)
    : type_(type), transform_(transform) {};
    __host__ __device__ ~Primitive() {};

    __host__ __device__ Primitive& Transform(const Eigen::Matrix4f& transform) { transform_ = transform_ * transform; return *this;};
    __host__ __device__
    virtual geometry::AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const { return geometry::AxisAlignedBoundingBox(); };

    PrimitiveType type_ = PrimitiveType::Unspecified;
    Eigen::Matrix4f transform_;
};

class Box : public Primitive {
public:
    __host__ __device__ Box() : Primitive(Primitive::PrimitiveType::Box) {};
    __host__ __device__ Box(const Eigen::Vector3f lengths) : Primitive(Primitive::PrimitiveType::Box), lengths_(lengths) {};
    __host__ __device__ Box(const Eigen::Vector3f lengths, const Eigen::Matrix4f& transform)
    : Primitive(Primitive::PrimitiveType::Sphere, transform), lengths_(lengths) {};
    __host__ __device__ ~Box() {};

    __host__ __device__
    geometry::AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const {
        const auto ra = transform_.block<3, 3>(0, 0) * 0.5 * lengths_;
        return geometry::AxisAlignedBoundingBox(-ra.array().abs().matrix() + transform_.block<3, 1>(0, 3),
                                                ra.array().abs().matrix() + transform_.block<3, 1>(0, 3));
    };

    Eigen::Vector3f lengths_ = Eigen::Vector3f::Zero();
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

    __host__ __device__
    geometry::AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const {
        return geometry::AxisAlignedBoundingBox(-Eigen::Vector3f(radius_, radius_, radius_) + transform_.block<3, 1>(0, 3),
                                                Eigen::Vector3f(radius_, radius_, radius_) + transform_.block<3, 1>(0, 3));
    };

    float radius_;
};

class Capsule : public Primitive {
public:
    __host__ __device__ Capsule() : Primitive(Primitive::PrimitiveType::Capsule), radius_(0), height_(0) {};
    __host__ __device__ Capsule(float radius, float height) : Primitive(Primitive::PrimitiveType::Capsule), radius_(radius), height_(height) {};
    __host__ __device__ Capsule(float radius, float height, const Eigen::Matrix4f& transform)
    : Primitive(Primitive::PrimitiveType::Capsule, transform), radius_(radius), height_(height) {};
    __host__ __device__ ~Capsule() {};

    __host__ __device__
    geometry::AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const {
        const Eigen::Vector3f pa = transform_.block<3, 3>(0, 0) * Eigen::Vector3f(0.0, 0.0, 0.5 * height_);
        const Eigen::Vector3f pb = -pa;
        const Eigen::Vector3f min_bound = (pa.array().min(pb.array()) - radius_).matrix() + transform_.block<3, 1>(0, 3);
        const Eigen::Vector3f max_bound = (pa.array().max(pb.array()) + radius_).matrix() + transform_.block<3, 1>(0, 3);
        return geometry::AxisAlignedBoundingBox(min_bound, max_bound);
    };

    float radius_;
    float height_;
};

union PrimitivePack {
    __host__ __device__ PrimitivePack() : primitive_() {};
    __host__ __device__ PrimitivePack(const PrimitivePack &other) {
        *this = other;
    };
    __host__ __device__ PrimitivePack &operator=(const PrimitivePack &other) {
        switch (other.primitive_.type_) {
            case Primitive::PrimitiveType::Box:
                box_ = other.box_;
                break;
            case Primitive::PrimitiveType::Sphere:
                sphere_ = other.sphere_;
                break;
            case Primitive::PrimitiveType::Capsule:
                capsule_ = other.capsule_;
                break;
            default:
                primitive_ = other.primitive_;
                break;
        }
        return *this;
    };
    __host__ __device__ ~PrimitivePack() {};
    Primitive primitive_;
    Box box_;
    Sphere sphere_;
    Capsule capsule_;
};

__host__ __device__
inline PrimitivePack operator+(const PrimitivePack &lhs, const PrimitivePack& rhs) {
    return lhs;
}

typedef utility::device_vector<PrimitivePack> PrimitiveArray;

std::shared_ptr<geometry::VoxelGrid> CreateVoxelGrid(const Primitive& primitive, float voxel_size);
std::shared_ptr<geometry::VoxelGrid> CreateVoxelGridWithSweeping(const Primitive& primitive, 
    float voxel_size, const Eigen::Matrix4f& dst, int sampling = 100);
std::shared_ptr<geometry::TriangleMesh> CreateTriangleMesh(const Primitive& primitive);

}
}