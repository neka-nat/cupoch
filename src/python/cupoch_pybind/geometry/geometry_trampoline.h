/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
**/
#pragma once

#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/geometry.h"
#include "cupoch/geometry/geometry_base.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch_pybind/cupoch_pybind.h"
#include "cupoch_pybind/geometry/geometry.h"

using namespace cupoch;

template <class GeometryT = geometry::Geometry>
class PyGeometry : public GeometryT {
public:
    using GeometryT::GeometryT;
    GeometryT& Clear() override {
        PYBIND11_OVERLOAD_PURE(GeometryT&, GeometryT, );
    }
    bool IsEmpty() const override { PYBIND11_OVERLOAD_PURE(bool, GeometryT, ); }
};

template <class Geometry3DBase = geometry::GeometryBase3D>
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

template <class Geometry2DBase = geometry::GeometryBase2D>
class PyGeometry2D : public PyGeometry<Geometry2DBase> {
public:
    using PyGeometry<Geometry2DBase>::PyGeometry;
    Eigen::Vector2f GetMinBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector2f, Geometry2DBase, );
    }
    Eigen::Vector2f GetMaxBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector2f, Geometry2DBase, );
    }
    Geometry2DBase& Transform(const Eigen::Matrix3f& transformation) override {
        PYBIND11_OVERLOAD_PURE(Geometry2DBase&, Geometry2DBase, transformation);
    }
};

template <class GeometryNoTrans2DBase = geometry::GeometryBaseNoTrans2D>
class PyGeometryNoTrans2D : public PyGeometry<GeometryNoTrans2DBase> {
public:
    using PyGeometry<GeometryNoTrans2DBase>::PyGeometry;
    Eigen::Vector2f GetMinBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector2f, GeometryNoTrans2DBase, );
    }
    Eigen::Vector2f GetMaxBound() const override {
        PYBIND11_OVERLOAD_PURE(Eigen::Vector2f, GeometryNoTrans2DBase, );
    }
};