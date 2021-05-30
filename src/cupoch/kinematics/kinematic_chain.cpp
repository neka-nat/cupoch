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
#include "cupoch/kinematics/kinematic_chain.h"

#include "cupoch/io/class_io/trianglemesh_io.h"
#include "cupoch/utility/helper.h"

namespace cupoch {
namespace kinematics {

namespace {
std::unordered_map<int, Joint::JointType> joint_type_map = {
        {urdf::Joint::UNKNOWN, Joint::JointType::Fixed},
        {urdf::Joint::FIXED, Joint::JointType::Fixed},
        {urdf::Joint::REVOLUTE, Joint::JointType::Revolute},
        {urdf::Joint::CONTINUOUS, Joint::JointType::Revolute},
        {urdf::Joint::PRISMATIC, Joint::JointType::Prismatic}};
}

Eigen::Matrix4f Frame::GetTransform(const float theta) const {
    Eigen::Matrix4f t = Eigen::Matrix4f::Identity();
    switch (joint_.type_) {
        case Joint::JointType::Revolute:
            t.block<3, 3>(0, 0) =
                    Eigen::AngleAxisf(theta, joint_.axis_).toRotationMatrix();
            break;
        case Joint::JointType::Prismatic:
            t.block<3, 1>(0, 3) = theta * joint_.axis_;
    }
    return joint_.offset_ * t;
}

KinematicChain& KinematicChain::BuildFromURDF(const std::string& filename) {
    std::vector<std::string> tokens;
    utility::SplitString(tokens, filename, "/", false);
    root_path_ = filename;
    utility::RightStripString(root_path_, tokens[tokens.size() - 1]);
    urdf::ModelInterfaceSharedPtr robot = urdf::parseURDFFile(filename);
    auto& lmap = robot->links_;
    auto& jmap = robot->joints_;
    std::vector<urdf::JointSharedPtr> joints;
    urdf::LinkSharedPtr root_link = nullptr;
    for (std::map<std::string, urdf::JointSharedPtr>::const_iterator itr =
                 jmap.begin();
         itr != jmap.end(); ++itr) {
        joints.push_back(itr->second);
    }
    int n_joints = joints.size();
    std::vector<bool> has_root(n_joints, true);
    for (int i = 0; i < n_joints; ++i) {
        for (int j = i + 1; j < n_joints; ++j) {
            if (joints[i]->parent_link_name == joints[j]->child_link_name) {
                has_root[i] = false;
            } else if (joints[j]->parent_link_name ==
                       joints[i]->child_link_name) {
                has_root[j] = false;
            }
        }
    }
    for (int i = 0; i < n_joints; ++i) {
        if (has_root[i]) {
            root_link = lmap[joints[i]->parent_link_name];
        }
    }
    root_ = Frame();
    root_.joint_ = Joint("root_joint");
    std::vector<ShapeInfo> collisions;
    for (const auto& col : root_link->collision_array) {
        collisions.push_back(ConvertCollision(col, root_path_));
    }
    std::vector<ShapeInfo> visuals;
    for (const auto& vis : root_link->visual_array) {
        visuals.push_back(ConvertVisual(vis, root_path_));
    }
    root_.link_ = Link(root_link->name, collisions, visuals);
    link_map_[root_.link_.name_] = &(root_.link_);
    root_.children_ = BuildChainRecurse(root_, lmap, joints);
    return *this;
}

KinematicChain::LinkPos KinematicChain::ForwardKinematics(
        const KinematicChain::JointMap& jmap,
        const Eigen::Matrix4f& base) const {
    return ForwardKinematicsRecurse(root_, jmap, base);
}

std::unordered_map<std::string, std::shared_ptr<const geometry::Geometry>>
KinematicChain::GetTransformedVisualGeometryMap(
        const KinematicChain::LinkPos& link_pos) const {
    std::unordered_map<std::string, std::shared_ptr<const geometry::Geometry>>
            ans;
    for (const auto& link : link_pos) {
        auto sub = std::make_shared<geometry::TriangleMesh>();
        for (const auto& visual : link_map_.at(link.first)->visuals_) {
            if (!visual.mesh_) {
                continue;
            }
            *sub += *visual.mesh_;
        }
        sub->Transform(link.second);
        ans.emplace(link.first, sub);
    }
    return ans;
}

std::vector<std::shared_ptr<Frame>> KinematicChain::BuildChainRecurse(
        Frame& frame,
        const std::map<std::string, urdf::LinkSharedPtr>& lmap,
        const std::vector<urdf::JointSharedPtr>& joints) {
    std::vector<std::shared_ptr<Frame>> children;
    for (const auto& joint : joints) {
        if (joint->parent_link_name == frame.link_.name_) {
            auto child = std::make_shared<Frame>();
            child->joint_ = Joint(
                    joint->name, joint_type_map[joint->type],
                    ConvertTransform(joint->parent_to_joint_origin_transform),
                    Eigen::Vector3f(joint->axis.x, joint->axis.y,
                                    joint->axis.z));
            auto link = lmap.at(joint->child_link_name);
            std::vector<ShapeInfo> collisions;
            for (const auto& col : link->collision_array) {
                collisions.push_back(ConvertCollision(col, root_path_));
            }
            std::vector<ShapeInfo> visuals;
            for (const auto& vis : link->visual_array) {
                visuals.push_back(ConvertVisual(vis, root_path_));
            }
            child->link_ = Link(link->name, collisions, visuals);
            link_map_[child->link_.name_] = &(child->link_);
            child->children_ = BuildChainRecurse(*child, lmap, joints);
            children.push_back(child);
        }
    }
    return children;
}

KinematicChain::LinkPos KinematicChain::ForwardKinematicsRecurse(
        const Frame& frame,
        const KinematicChain::JointMap& jmap,
        const Eigen::Matrix4f& base) const {
    KinematicChain::LinkPos ans;
    auto jth = jmap.find(frame.joint_.name_);
    ans[frame.link_.name_] =
            base * ((jth != jmap.end()) ? frame.GetTransform(jth->second)
                                        : frame.joint_.offset_);
    for (const auto& child : frame.children_) {
        auto sub =
                ForwardKinematicsRecurse(*child, jmap, ans[frame.link_.name_]);
        for (KinematicChain::LinkPos::const_iterator jtr = sub.begin();
             jtr != sub.end(); ++jtr) {
            ans[jtr->first] = jtr->second;
        }
    }
    return ans;
}

Eigen::Matrix4f ConvertTransform(const urdf::Pose& pose) {
    Eigen::Matrix4f ans = Eigen::Matrix4f::Identity();
    double x, y, z, w;
    pose.rotation.getQuaternion(x, y, z, w);
    Eigen::Quaternionf q(w, x, y, z);
    ans.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    ans.block<3, 1>(0, 3) =
            Eigen::Vector3f(pose.position.x, pose.position.y, pose.position.z);
    return ans;
}

ShapeInfo ConvertGeometry(const urdf::GeometrySharedPtr& geometry,
                          const urdf::Pose& pose,
                          const std::string& root_path) {
    if (!geometry) {
        return ShapeInfo(nullptr, nullptr);
    }
    switch (geometry->type) {
        case urdf::Geometry::SPHERE: {
            urdf::SphereSharedPtr sphere =
                    std::dynamic_pointer_cast<urdf::Sphere>(geometry);
            return ShapeInfo(std::make_shared<collision::Sphere>(
                    sphere->radius, ConvertTransform(pose).block<3, 1>(0, 3)));
        }
        case urdf::Geometry::BOX: {
            urdf::BoxSharedPtr box =
                    std::dynamic_pointer_cast<urdf::Box>(geometry);
            return ShapeInfo(std::make_shared<collision::Box>(
                    Eigen::Vector3f(box->dim.x, box->dim.y, box->dim.z),
                    ConvertTransform(pose)));
        }
        case urdf::Geometry::CYLINDER: {
            urdf::CylinderSharedPtr cylinder =
                    std::dynamic_pointer_cast<urdf::Cylinder>(geometry);
            return ShapeInfo(std::make_shared<collision::Cylinder>(
                    cylinder->radius, cylinder->length,
                    ConvertTransform(pose)));
        }
        case urdf::Geometry::MESH: {
            urdf::MeshSharedPtr mesh =
                    std::dynamic_pointer_cast<urdf::Mesh>(geometry);
            std::string tmp = mesh->filename;
            const std::string p = "package://";
            auto r = std::search(tmp.begin(), tmp.end(), p.begin(), p.end());
            if (r != tmp.end()) {
                tmp.replace(std::distance(tmp.begin(), r), p.length(), "");
            }
            auto tri = io::CreateMeshFromFile(root_path + tmp);
            auto trans = ConvertTransform(pose);
            tri->Transform(trans);
            return ShapeInfo(std::make_shared<collision::Mesh>(trans), tri);
        }
        default: {
            return ShapeInfo(nullptr, nullptr);
        }
    }
}

ShapeInfo ConvertCollision(const urdf::CollisionSharedPtr& collision,
                           const std::string& root_path) {
    if (!collision) {
        return ShapeInfo(nullptr, nullptr);
    }
    return ConvertGeometry(collision->geometry, collision->origin, root_path);
}

ShapeInfo ConvertVisual(const urdf::VisualSharedPtr& visual,
                        const std::string& root_path) {
    if (!visual) {
        return ShapeInfo(nullptr, nullptr);
    }
    return ConvertGeometry(visual->geometry, visual->origin, root_path);
}

}  // namespace kinematics
}  // namespace cupoch