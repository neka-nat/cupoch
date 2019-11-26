#pragma once

namespace cupoc {
namespace geometry {

class Geometry {
public:
    enum class GeometryType {
        Unspecified = 0,
        PointCloud = 1,
    };

public:
    virtual ~Geometry() {}

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
}  // namespace cupoc