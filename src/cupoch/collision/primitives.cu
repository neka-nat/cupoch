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
#include "cupoch/collision/primitives.h"
#include "cupoch/geometry/intersection_test.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace collision {

namespace {

template <class T>
struct create_from_primitive_functor {
    create_from_primitive_functor(const T& primitive,
                                  float voxel_size,
                                  int num_w,
                                  int num_h,
                                  int num_d)
        : primitive_(primitive),
          voxel_size_(voxel_size),
          box_half_size_(Eigen::Vector3f(
                  voxel_size / 2, voxel_size / 2, voxel_size / 2)),
          num_w_(num_w),
          num_h_(num_h),
          num_d_(num_d){};
    const T primitive_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const int num_w_;
    const int num_h_;
    const int num_d_;
    __device__ virtual bool intersect(
            const Eigen::Vector3f& box_center) const = 0;
    __device__ thrust::tuple<Eigen::Vector3i, geometry::Voxel> operator()(
            size_t idx) const {
        int widx = idx / (num_h_ * num_d_) - num_w_ / 2;
        int hdidx = idx % (num_h_ * num_d_);
        int hidx = hdidx / num_d_ - num_h_ / 2;
        int didx = hdidx % num_d_ - num_d_ / 2;

        const Eigen::Vector3f box_center =
                Eigen::Vector3f(widx + 0.5, hidx + 0.5, didx + 0.5) *
                voxel_size_;
        if (intersect(box_center)) {
            Eigen::Vector3i grid_index(widx, hidx, didx);
            return thrust::make_tuple(grid_index, geometry::Voxel(grid_index));
        }
        return thrust::make_tuple(
                Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX,
                                geometry::INVALID_VOXEL_INDEX,
                                geometry::INVALID_VOXEL_INDEX),
                geometry::Voxel());
    }
};

struct create_from_box_functor : create_from_primitive_functor<Box> {
    using create_from_primitive_functor<Box>::create_from_primitive_functor<
            Box>;
    __device__ bool intersect(const Eigen::Vector3f& box_center) const {
        return geometry::intersection_test::BoxBox(
                0.5 * primitive_.lengths_,
                primitive_.transform_.block<3, 3>(0, 0),
                Eigen::Vector3f::Zero(), box_half_size_,
                Eigen::Matrix3f::Identity(), box_center);
    }
};

struct create_from_sphere_functor : create_from_primitive_functor<Sphere> {
    using create_from_primitive_functor<Sphere>::create_from_primitive_functor<
            Sphere>;
    __device__ bool intersect(const Eigen::Vector3f& box_center) const {
        return geometry::intersection_test::SphereAABB(
                Eigen::Vector3f::Zero(), primitive_.radius_,
                box_center - box_half_size_, box_center + box_half_size_);
    }
};

struct create_from_capsule_functor : create_from_primitive_functor<Capsule> {
    using create_from_primitive_functor<Capsule>::create_from_primitive_functor<
            Capsule>;
    __device__ bool intersect(const Eigen::Vector3f& box_center) const {
        return geometry::intersection_test::CapsuleAABB(
                primitive_.radius_,
                Eigen::Vector3f(0.0, 0.0, -primitive_.height_ / 2),
                Eigen::Vector3f(0.0, 0.0, primitive_.height_),
                box_center - box_half_size_, box_center + box_half_size_);
    }
};

template <class T>
struct create_from_swept_primitive_functor {
    create_from_swept_primitive_functor(const T& primitive,
                                        float voxel_size,
                                        int num_w,
                                        int num_h,
                                        int num_d,
                                        const Eigen::Matrix4f& dst,
                                        int sampling)
        : primitive_(primitive),
          voxel_size_(voxel_size),
          box_half_size_(Eigen::Vector3f(
                  voxel_size / 2, voxel_size / 2, voxel_size / 2)),
          num_w_(num_w),
          num_h_(num_h),
          num_d_(num_d),
          dst_(dst),
          sampling_(sampling){};
    const T primitive_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const int num_w_;
    const int num_h_;
    const int num_d_;
    const Eigen::Matrix4f dst_;
    const int sampling_;
    __device__ virtual bool intersect(
            const Eigen::Matrix4f& primitive_trans,
            const Eigen::Vector3f& box_center) const = 0;
    __device__ thrust::tuple<Eigen::Vector3i, geometry::Voxel> operator()(
            size_t idx) const {
        int widx = idx / (num_h_ * num_d_ * sampling_) - num_w_ / 2;
        int hdsidx = idx % (num_h_ * num_d_ * sampling_);
        int hidx = hdsidx / (num_d_ * sampling_) - num_h_ / 2;
        int dsidx = hdsidx % (num_d_ * sampling_);
        int didx = dsidx / sampling_ - num_d_ / 2;
        int sidx = dsidx % sampling_;

        const float ratio = float(sidx) / sampling_;
        Eigen::Quaternionf qdst(dst_.block<3, 3>(0, 0));
        Eigen::Quaternionf tq =
                qdst.slerp(1.0 - ratio, Eigen::Quaternionf::Identity());
        Eigen::Matrix4f dtf = dst_;
        dtf.block<3, 1>(0, 3) *= ratio;
        dtf.block<3, 3>(0, 0) = tq.toRotationMatrix();
        Eigen::Vector3f box_center =
                (dtf * Eigen::Vector4f(voxel_size_ * (widx + 0.5),
                                       voxel_size_ * (hidx + 0.5),
                                       voxel_size_ * (didx + 0.5), 1.0)).template head<3>();
        Eigen::Vector3i grid_index = (box_center / voxel_size_).template cast<int>();
        box_center = grid_index.cast<float>() * voxel_size_;
        if (intersect(dtf, box_center)) {
            return thrust::make_tuple(grid_index, geometry::Voxel(grid_index));
        }
        return thrust::make_tuple(
                Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX,
                                geometry::INVALID_VOXEL_INDEX,
                                geometry::INVALID_VOXEL_INDEX),
                geometry::Voxel());
    }
};

struct create_from_swept_box_functor
    : create_from_swept_primitive_functor<Box> {
    using create_from_swept_primitive_functor<
            Box>::create_from_swept_primitive_functor<Box>;
    __device__ bool intersect(const Eigen::Matrix4f& primitive_trans,
                              const Eigen::Vector3f& box_center) const {
        Eigen::Matrix4f rot_pmtv = primitive_.transform_;
        rot_pmtv.block<3, 1>(0, 3) = Eigen::Vector3f::Zero();
        const Eigen::Matrix4f t = primitive_trans * rot_pmtv;
        return geometry::intersection_test::BoxBox(
                0.5 * primitive_.lengths_, t.block<3, 3>(0, 0),
                t.block<3, 1>(0, 3), box_half_size_,
                Eigen::Matrix3f::Identity(), box_center);
    }
};

struct create_from_swept_sphere_functor
    : create_from_swept_primitive_functor<Sphere> {
    using create_from_swept_primitive_functor<
            Sphere>::create_from_swept_primitive_functor<Sphere>;
    __device__ bool intersect(const Eigen::Matrix4f& primitive_trans,
                              const Eigen::Vector3f& box_center) const {
        return geometry::intersection_test::SphereAABB(
                primitive_trans.block<3, 1>(0, 3), primitive_.radius_,
                box_center - box_half_size_, box_center + box_half_size_);
    }
};

struct create_from_swept_capsule_functor
    : create_from_swept_primitive_functor<Capsule> {
    using create_from_swept_primitive_functor<
            Capsule>::create_from_swept_primitive_functor<Capsule>;
    __device__ bool intersect(const Eigen::Matrix4f& primitive_trans,
                              const Eigen::Vector3f& box_center) const {
        Eigen::Matrix4f rot_pmtv = primitive_.transform_;
        rot_pmtv.block<3, 1>(0, 3) = Eigen::Vector3f::Zero();
        const Eigen::Matrix4f t = primitive_trans * rot_pmtv;
        return geometry::intersection_test::CapsuleAABB(
                primitive_.radius_,
                -0.5 * primitive_.height_ * t.block<3, 1>(0, 2),
                primitive_.height_ * t.block<3, 1>(0, 2),
                box_center - box_half_size_, box_center + box_half_size_);
    }
};

template <typename Func>
void TransformAndResizeVoxel(geometry::VoxelGrid& voxelgrid,
                             size_t n_total,
                             Func fn) {
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      make_tuple_iterator(voxelgrid.voxels_keys_.begin(),
                                          voxelgrid.voxels_values_.begin()),
                      fn);
    auto remove_fn =
            [] __device__(
                    const thrust::tuple<Eigen::Vector3i, geometry::Voxel>& x)
            -> bool {
        Eigen::Vector3i idxs = thrust::get<0>(x);
        return idxs == Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX,
                                       geometry::INVALID_VOXEL_INDEX,
                                       geometry::INVALID_VOXEL_INDEX);
    };
    remove_if_vectors(remove_fn, voxelgrid.voxels_keys_,
                      voxelgrid.voxels_values_);
}

template <class T, class Func>
std::shared_ptr<geometry::VoxelGrid> CreateVoxelGridFromPrimitive(
        const T& primitive, float voxel_size) {
    auto output = std::make_shared<geometry::VoxelGrid>();
    if (voxel_size <= 0.0) {
        utility::LogError("[CreateVoxelGrid] voxel_size <= 0.");
        return output;
    }
    const Eigen::Vector3f voxel_size3(voxel_size, voxel_size, voxel_size);
    output->voxel_size_ = voxel_size;
    output->origin_ = primitive.transform_.template block<3, 1>(0, 3);
    auto bbx = primitive.GetAxisAlignedBoundingBox();
    const Eigen::Vector3f min_bound =
            bbx.min_bound_ - output->origin_ - voxel_size3 * 0.5;
    const Eigen::Vector3f max_bound =
            bbx.max_bound_ - output->origin_ + voxel_size3 * 0.5;

    Eigen::Vector3f grid_size = max_bound - min_bound;
    int num_w = int(std::round(grid_size(0) / voxel_size));
    int num_h = int(std::round(grid_size(1) / voxel_size));
    int num_d = int(std::round(grid_size(2) / voxel_size));
    size_t n_total = num_w * num_h * num_d;
    Func func(primitive, voxel_size, num_w, num_h, num_d);
    output->voxels_keys_.resize(n_total);
    output->voxels_values_.resize(n_total);
    TransformAndResizeVoxel(*output, n_total, func);
    return output;
}

template <class T, class Func>
std::shared_ptr<geometry::VoxelGrid> CreateVoxelGridWithSweepingFromPrimitive(
        const T& primitive,
        float voxel_size,
        const Eigen::Matrix4f& dst,
        int sampling) {
    auto output = std::make_shared<geometry::VoxelGrid>();
    if (voxel_size <= 0.0) {
        utility::LogError("[CreateVoxelGrid] voxel_size <= 0.");
        return output;
    }
    const Eigen::Vector3f voxel_size3(voxel_size, voxel_size, voxel_size);
    auto bbx = primitive.GetAxisAlignedBoundingBox();
    const Eigen::Vector3f min_bound =
            bbx.min_bound_ - output->origin_ - voxel_size3 * 0.5;
    const Eigen::Vector3f max_bound =
            bbx.max_bound_ - output->origin_ + voxel_size3 * 0.5;
    output->voxel_size_ = voxel_size;
    output->origin_ = primitive.transform_.template block<3, 1>(0, 3);

    Eigen::Vector3f grid_size = max_bound - min_bound;
    int num_w = int(std::round(grid_size(0) / voxel_size));
    int num_h = int(std::round(grid_size(1) / voxel_size));
    int num_d = int(std::round(grid_size(2) / voxel_size));
    size_t n_total = num_w * num_h * num_d * sampling;
    Func func(primitive, voxel_size, num_w, num_h, num_d, dst, sampling);
    output->voxels_keys_.resize(n_total);
    output->voxels_values_.resize(n_total);
    TransformAndResizeVoxel(*output, n_total, func);
    return output;
}

}  // namespace

std::shared_ptr<geometry::VoxelGrid> CreateVoxelGrid(const Primitive& primitive,
                                                     float voxel_size) {
    switch (primitive.type_) {
        case Primitive::PrimitiveType::Box: {
            const Box& box = (const Box&)primitive;
            return CreateVoxelGridFromPrimitive<Box, create_from_box_functor>(
                    box, voxel_size);
        }
        case Primitive::PrimitiveType::Sphere: {
            const Sphere& sphere = (const Sphere&)primitive;
            return CreateVoxelGridFromPrimitive<Sphere,
                                                create_from_sphere_functor>(
                    sphere, voxel_size);
        }
        case Primitive::PrimitiveType::Capsule: {
            const Capsule& capsule = (const Capsule&)primitive;
            return CreateVoxelGridFromPrimitive<Capsule,
                                                create_from_capsule_functor>(
                    capsule, voxel_size);
        }
        default: {
            utility::LogError("[CreateVoxelGrid] Unsupported primitive type.");
            return std::shared_ptr<geometry::VoxelGrid>();
        }
    }
}

std::shared_ptr<geometry::VoxelGrid> CreateVoxelGridWithSweeping(
        const Primitive& primitive,
        float voxel_size,
        const Eigen::Matrix4f& dst,
        int sampling) {
    switch (primitive.type_) {
        case Primitive::PrimitiveType::Box: {
            const Box& box = (const Box&)primitive;
            return CreateVoxelGridWithSweepingFromPrimitive<
                    Box, create_from_swept_box_functor>(box, voxel_size, dst,
                                                        sampling);
        }
        case Primitive::PrimitiveType::Sphere: {
            const Sphere& sphere = (const Sphere&)primitive;
            return CreateVoxelGridWithSweepingFromPrimitive<
                    Sphere, create_from_swept_sphere_functor>(
                    sphere, voxel_size, dst, sampling);
        }
        case Primitive::PrimitiveType::Capsule: {
            const Capsule& capsule = (const Capsule&)primitive;
            return CreateVoxelGridWithSweepingFromPrimitive<
                    Capsule, create_from_swept_capsule_functor>(
                    capsule, voxel_size, dst, sampling);
        }
        default: {
            utility::LogError(
                    "[CreateVoxelGridWithSweeping] Unsupported primitive "
                    "type.");
            return std::shared_ptr<geometry::VoxelGrid>();
        }
    }
}

std::shared_ptr<geometry::TriangleMesh> CreateTriangleMesh(
        const Primitive& primitive) {
    switch (primitive.type_) {
        case Primitive::PrimitiveType::Box: {
            const Box& box = (const Box&)primitive;
            auto output = geometry::TriangleMesh::CreateBox(
                    box.lengths_[0], box.lengths_[1], box.lengths_[2]);
            Eigen::Matrix4f tf = primitive.transform_;
            tf.topRightCorner<3, 1>() -= box.lengths_ * 0.5;
            output->Transform(primitive.transform_);
            return output;
        }
        case Primitive::PrimitiveType::Sphere: {
            const Sphere& sphere = (const Sphere&)primitive;
            auto output = geometry::TriangleMesh::CreateSphere(sphere.radius_);
            output->Transform(primitive.transform_);
            return output;
        }
        case Primitive::PrimitiveType::Capsule: {
            const Capsule& capsule = (const Capsule&)primitive;
            auto output = geometry::TriangleMesh::CreateCylinder(
                    capsule.radius_, capsule.height_);
            output->Transform(primitive.transform_);
            return output;
        }
        default: {
            utility::LogError(
                    "[CreateTriangleMesh] Unsupported primitive type.");
            return std::make_shared<geometry::TriangleMesh>();
        }
    }
}

}  // namespace collision
}  // namespace cupoch