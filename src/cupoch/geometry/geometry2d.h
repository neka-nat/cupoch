#pragma once

#include "cupoch/geometry/geometry.h"
#include <Eigen/Core>

namespace cupoch {
namespace geometry {

/// \class Geometry2D
///
/// \brief The base geometry class for 2D geometries.
///
/// Main class for 2D geometries, Derives all data from Geometry Base class.
class Geometry2D : public Geometry {
public:
    virtual ~Geometry2D() {}

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type  type of object based on GeometryType
    Geometry2D(GeometryType type) : Geometry(type, 2) {}

public:
    Geometry& Clear() override = 0;
    bool IsEmpty() const override = 0;
    /// Returns min bounds for geometry coordinates.
    virtual Eigen::Vector2f GetMinBound() const = 0;
    /// Returns max bounds for geometry coordinates.
    virtual Eigen::Vector2f GetMaxBound() const = 0;
};

}
} 
