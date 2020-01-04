#pragma once

#include "cupoch/geometry/bounding_volume.h"
#include "cupoch/geometry/geometry.h"
#include "cupoch/geometry/geometry2d.h"
#include "cupoch/geometry/geometry3d.h"
#include "cupoch/geometry/trianglemesh.h"

#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/cupoch_pybind.h"

using namespace cupoch;

template <class GeometryBase = geometry::Geometry>
class PyGeometry : public GeometryBase {
public:
    using GeometryBase::GeometryBase;
    GeometryBase& Clear() override {
        PYBIND11_OVERLOAD_PURE(GeometryBase&, GeometryBase, );
    }
    bool IsEmpty() const override {
        PYBIND11_OVERLOAD_PURE(bool, GeometryBase, );
    }
};

template <class Geometry3DBase = geometry::Geometry3D>
class PyGeometry3D : public PyGeometry<Geometry3DBase> {
public:
    using PyGeometry<Geometry3DBase>::PyGeometry;
    Eigen::Vector3f GetMinBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector3f, Geometry3DBase, );
    }
    Eigen::Vector3f GetMaxBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector3f, Geometry3DBase, );
    }
    Eigen::Vector3f GetCenter() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector3f, Geometry3DBase, );
    }
    geometry::AxisAlignedBoundingBox GetAxisAlignedBoundingBox()
            const override {
        PYBIND11_OVERLOAD_PURE(geometry::AxisAlignedBoundingBox,
                               Geometry3DBase, );
    }
    Geometry3DBase& Transform(const Eigen::Matrix4f& transformation) override {
        PYBIND11_OVERLOAD_PURE(Geometry3DBase&, Geometry3DBase, transformation);
    }
};

template <class Geometry2DBase = geometry::Geometry2D>
class PyGeometry2D : public PyGeometry<Geometry2DBase> {
public:
    using PyGeometry<Geometry2DBase>::PyGeometry;
    Eigen::Vector2f GetMinBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector2f, Geometry2DBase, );
    }
    Eigen::Vector2f GetMaxBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector2f, Geometry2DBase, );
    }
};