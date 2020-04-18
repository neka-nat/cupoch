#include "cupoch/collision/primitives.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/geometry/intersection_test.h"

#include "cupoch/utility/eigen.h"

namespace cupoch {
namespace collision {

namespace {

struct create_from_sphere_functor {
    create_from_sphere_functor(float radius,
                               float voxel_size,
                               int num_w,
                               int num_h,
                               int num_d)
        : radius_(radius),
          voxel_size_(voxel_size),
          box_half_size_(Eigen::Vector3f(
                  voxel_size / 2, voxel_size / 2, voxel_size / 2)),
          num_w_(num_w),
          num_h_(num_h),
          num_d_(num_d) {};
    const float radius_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const int num_w_;
    const int num_h_;
    const int num_d_;
    __device__ thrust::tuple<Eigen::Vector3i, geometry::Voxel> operator()(
            size_t idx) const {
        int widx = idx / (num_h_ * num_d_) - num_w_ / 2;
        int hdidx = idx % (num_h_ * num_d_);
        int hidx = hdidx / num_d_ - num_h_ / 2;
        int didx = hdidx % num_d_ - num_d_ / 2;

        const Eigen::Vector3f box_center = Eigen::Vector3f(widx, hidx, didx) * voxel_size_;
        if (geometry::intersection_test::SphereAABB(Eigen::Vector3f::Zero(), radius_,
                                                    box_center - box_half_size_,
                                                    box_center + box_half_size_)) {
            Eigen::Vector3i grid_index(widx, hidx, didx);
            return thrust::make_tuple(grid_index,
                                      geometry::Voxel(grid_index));
        }
        return thrust::make_tuple(Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX,
                                                  geometry::INVALID_VOXEL_INDEX,
                                                  geometry::INVALID_VOXEL_INDEX),
                                  geometry::Voxel());
    }
};

struct create_from_swept_sphere_functor {
    create_from_swept_sphere_functor(float radius,
                                    float voxel_size,
                                    int num_w,
                                    int num_h,
                                    int num_d,
                                    const Eigen::Vector3f& step,
                                    int sampling)
        : radius_(radius),
          voxel_size_(voxel_size),
          box_half_size_(Eigen::Vector3f(
                  voxel_size / 2, voxel_size / 2, voxel_size / 2)),
          num_w_(num_w),
          num_h_(num_h),
          num_d_(num_d),
          step_(step),
          sampling_(sampling) {};
    const float radius_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const int num_w_;
    const int num_h_;
    const int num_d_;
    const Eigen::Vector3f step_;
    const int sampling_;
    __device__ thrust::tuple<Eigen::Vector3i, geometry::Voxel> operator()(
            size_t idx) const {
        int widx = idx / (num_h_ * num_d_ * sampling_)  - num_w_ / 2;
        int hdsidx = idx % (num_h_ * num_d_ * sampling_);
        int hidx = hdsidx / (num_d_ * sampling_) - num_h_ / 2;
        int dsidx = hdsidx % (num_d_ * sampling_);
        int didx = dsidx / sampling_ - num_d_ / 2;
        int sidx = dsidx % sampling_;

        Eigen::Vector3f diff = sidx * step_;
        Eigen::Vector3i offset = (Eigen::device_floor<Eigen::Vector3f>(diff / voxel_size_)).cast<int>();
        const Eigen::Vector3f box_center = (offset.cast<float>() + Eigen::Vector3f(widx, hidx, didx)) * voxel_size_;
        if (geometry::intersection_test::SphereAABB(sidx * step_, radius_,
                                                    box_center - box_half_size_,
                                                    box_center + box_half_size_)) {
            Eigen::Vector3i grid_index = offset + Eigen::Vector3i(widx, hidx, didx);
            return thrust::make_tuple(grid_index,
                                      geometry::Voxel(grid_index));
        }
        return thrust::make_tuple(Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX,
                                                  geometry::INVALID_VOXEL_INDEX,
                                                  geometry::INVALID_VOXEL_INDEX),
                                  geometry::Voxel());
    }
};

template <typename Func>
void TransformAndResizeVoxel(geometry::VoxelGrid& voxelgrid, size_t n_total, Func fn) {
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      make_tuple_iterator(voxelgrid.voxels_keys_.begin(),
                                          voxelgrid.voxels_values_.begin()),
                      fn);
    auto begin = make_tuple_iterator(voxelgrid.voxels_keys_.begin(),
                                     voxelgrid.voxels_values_.begin());
    auto end = thrust::remove_if(
            begin,
            make_tuple_iterator(voxelgrid.voxels_keys_.end(),
                                voxelgrid.voxels_values_.end()),
            [] __device__(
                    const thrust::tuple<Eigen::Vector3i, geometry::Voxel> &x)
                    -> bool {
                Eigen::Vector3i idxs = thrust::get<0>(x);
                return idxs == Eigen::Vector3i(geometry::INVALID_VOXEL_INDEX,
                                               geometry::INVALID_VOXEL_INDEX,
                                               geometry::INVALID_VOXEL_INDEX);
            });
    size_t n_out = thrust::distance(begin, end);
    voxelgrid.voxels_keys_.resize(n_out);
    voxelgrid.voxels_values_.resize(n_out);
}

}

Primitive::Primitive(Primitive::PrimitiveType type) : type_(type), transform_(Eigen::Matrix4f::Identity()) {};
Primitive::~Primitive() {}

Sphere::Sphere() : Primitive(Primitive::PrimitiveType::Sphere), radius_(0.0) {}
Sphere::Sphere(float radius) : Primitive(Primitive::PrimitiveType::Sphere), radius_(radius) {}
Sphere::Sphere(float radius, const Eigen::Vector3f& center)
    : Primitive(Primitive::PrimitiveType::Sphere), radius_(radius) {
    transform_.block<3, 1>(0, 3) = center;
}

Sphere::~Sphere() {}

std::shared_ptr<geometry::VoxelGrid> Sphere::CreateVoxelGrid(float voxel_size) const {
    auto output = std::make_shared<geometry::VoxelGrid>();
    if (radius_ <= 0.0) {
        utility::LogError("[CreateVoxelGrid] radius <= 0.");
    }
    if (voxel_size <= 0.0) {
        utility::LogError("[CreateVoxelGrid] voxel_size <= 0.");
    }
    const Eigen::Vector3f voxel_size3(voxel_size, voxel_size, voxel_size);
    const Eigen::Vector3f radius3(radius_, radius_, radius_);
    const Eigen::Vector3f min_bound = -radius3 - voxel_size3 * 0.5;
    const Eigen::Vector3f max_bound = radius3 + voxel_size3 * 0.5;
    output->voxel_size_ = voxel_size;
    output->origin_ = transform_.block<3, 1>(0, 3);

    Eigen::Vector3f grid_size = max_bound - min_bound;
    int num_w = int(std::round(grid_size(0) / voxel_size));
    int num_h = int(std::round(grid_size(1) / voxel_size));
    int num_d = int(std::round(grid_size(2) / voxel_size));
    size_t n_total = num_w * num_h * num_d;
    create_from_sphere_functor func(radius_, voxel_size, num_w, num_h, num_d);
    output->voxels_keys_.resize(n_total);
    output->voxels_values_.resize(n_total);
    TransformAndResizeVoxel(*output, n_total, func);
    return output;
}

std::shared_ptr<geometry::VoxelGrid> Sphere::CreateVoxelGridWithSweeping(
    float voxel_size, const Eigen::Matrix4f& dst, int sampling) const {
    auto output = std::make_shared<geometry::VoxelGrid>();
    if (radius_ <= 0.0) {
        utility::LogError("[CreateVoxelGrid] radius <= 0.");
    }
    if (voxel_size <= 0.0) {
        utility::LogError("[CreateVoxelGrid] voxel_size <= 0.");
    }
    const Eigen::Vector3f diff = dst.block<3, 1>(0, 3) - transform_.block<3, 1>(0, 3);
    const Eigen::Vector3f voxel_size3(voxel_size, voxel_size, voxel_size);
    const Eigen::Vector3f radius3(radius_, radius_, radius_);
    const Eigen::Vector3f min_bound_st = -radius3 - voxel_size3 * 0.5;
    const Eigen::Vector3f max_bound_st = radius3 + voxel_size3 * 0.5;
    const Eigen::Vector3f min_bound_en = diff + min_bound_st;
    const Eigen::Vector3f max_bound_en = diff + max_bound_st;
    const Eigen::Vector3f min_bound = min_bound_st.array().min(min_bound_en.array()).matrix();
    const Eigen::Vector3f max_bound = max_bound_st.array().max(max_bound_en.array()).matrix();
    output->voxel_size_ = voxel_size;
    output->origin_ = transform_.block<3, 1>(0, 3);

    Eigen::Vector3f grid_size = max_bound - min_bound;
    int num_w = int(std::round(grid_size(0) / voxel_size));
    int num_h = int(std::round(grid_size(1) / voxel_size));
    int num_d = int(std::round(grid_size(2) / voxel_size));
    size_t n_total = num_w * num_h * num_d * sampling;
    create_from_swept_sphere_functor func(radius_, voxel_size, num_w, num_h, num_d,
                                          diff / (sampling - 1), sampling);
    output->voxels_keys_.resize(n_total);
    output->voxels_values_.resize(n_total);
    TransformAndResizeVoxel(*output, n_total, func);
    return output;
}

std::shared_ptr<geometry::TriangleMesh> Sphere::CreateTriangleMesh() const {
    auto output = geometry::TriangleMesh::CreateSphere();
    output->Transform(transform_);
    return output;
}

}
}