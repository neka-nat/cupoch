#pragma once

#if !defined(__CUDACC__)
#if !defined(__host__)
#define __host__
#endif
#if !defined(__device__)
#define __device__
#endif
#endif

namespace cupoch {
namespace geometry {

class Geometry {
public:
    enum class GeometryType {
        Unspecified = 0,
        /// PointCloud
        PointCloud = 1,
        /// LineSet
        LineSet = 4,
        /// MeshBase
        MeshBase = 5,
        /// TriangleMesh
        TriangleMesh = 6,
        /// Image
        Image = 8,
        /// OrientedBoundingBox
        OrientedBoundingBox = 11,
        /// AxisAlignedBoundingBox
        AxisAlignedBoundingBox = 12,
    };

public:
    __host__ __device__
    ~Geometry() {} // non-virtual

protected:
    Geometry(GeometryType type, int dimension)
        : geometry_type_(type), dimension_(dimension) {}

public:
    virtual Geometry& Clear() = 0;
    virtual bool IsEmpty() const = 0;
    GeometryType GetGeometryType() const { return geometry_type_; }
    int Dimension() const { return dimension_; }

private:
    GeometryType geometry_type_ = GeometryType::Unspecified;
    int dimension_ = 3;
};

}  // namespace geometry
}  // namespace cupoch