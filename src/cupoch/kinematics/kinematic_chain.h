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
#include "cupoch/collision/primitives.h"
#include <urdf_parser/urdf_parser.h>

#include <vector>
#include <unordered_map>

namespace cupoch {
namespace kinematics {

class ShapeInfo {
public:
    ShapeInfo(std::shared_ptr<collision::Primitive> primitive = nullptr,
              std::shared_ptr<geometry::TriangleMesh> mesh = nullptr)
    : primitive_(primitive), mesh_(mesh) {
        if (primitive_ && !mesh_) {
            mesh_ = collision::CreateTriangleMesh(*primitive_);
        }
    };
    ~ShapeInfo() {};

    std::shared_ptr<collision::Primitive> primitive_;
    std::shared_ptr<geometry::TriangleMesh> mesh_;
};

class Link {
public:
    Link(const std::string& name = "") : name_(name) {};
    Link(const std::string& name,
         const ShapeInfo& collision,
         const ShapeInfo& visual)
    : name_(name), collisions_(1, collision), visuals_(1, visual) {};

    std::string name_;
    std::vector<ShapeInfo> collisions_;
    std::vector<ShapeInfo> visuals_;
};

class Joint {
public:
    enum JointType {
        Fixed = 0,
        Revolute = 1,
        Prismatic = 2,
    };

    Joint(const std::string& name = "") : name_(name) {};
    Joint(const std::string& name, JointType type, const Eigen::Matrix4f& offset, const Eigen::Vector3f& axis)
    : name_(name), type_(type), offset_(offset), axis_(axis) {};
    std::string name_;
    JointType type_;
    Eigen::Matrix4f offset_ = Eigen::Matrix4f::Identity();
    Eigen::Vector3f axis_;
};

class Frame {
public:
    Frame() {};
    ~Frame() {};

    Eigen::Matrix4f GetTransform(const float theta = 0.0) const;

    Link link_;
    Joint joint_;
    std::vector<std::shared_ptr<Frame>> children_;
};

class KinematicChain {
public:
    typedef std::unordered_map<std::string, float> JointMap;
    typedef std::unordered_map<std::string, Eigen::Matrix4f> LinkPos;

    KinematicChain(const std::string& filename = "") { if (!filename.empty()) BuildFromURDF(filename); };
    ~KinematicChain() {};
    KinematicChain& BuildFromURDF(const std::string& filename);

    LinkPos ForwardKinematics(const JointMap& jmap = JointMap(),
                              const Eigen::Matrix4f& base = Eigen::Matrix4f::Identity()) const;

    std::unordered_map<std::string, std::shared_ptr<const geometry::Geometry>>
    GetTransformedVisualGeometryMap(const LinkPos& link_pos) const;
private:
    std::vector<std::shared_ptr<Frame>> BuildChainRecurse(
        Frame& frame,
        const std::map<std::string, urdf::LinkSharedPtr>& lmap,
        const std::vector<urdf::JointSharedPtr>& joints);

    LinkPos ForwardKinematicsRecurse(const Frame& frame, const JointMap& jmap,
                                     const Eigen::Matrix4f& base) const;
public:
    Frame root_;
    std::string root_path_ = "";
    std::unordered_map<std::string, Link*> link_map_;
};

Eigen::Matrix4f ConvertTransform(const urdf::Pose& pose);
ShapeInfo ConvertGeometry(const urdf::GeometrySharedPtr& geometry, const urdf::Pose& pose, const std::string& root_path = "");
ShapeInfo ConvertCollision(const urdf::CollisionSharedPtr& collision, const std::string& root_path = "");
ShapeInfo ConvertVisual(const urdf::VisualSharedPtr& visual, const std::string& root_path = "");

}
}