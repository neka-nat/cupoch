#include <thrust/iterator/discard_iterator.h>

#include <numeric>

#include "cupoch/geometry/intersection_test.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/helper.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct create_dense_functor {
    create_dense_functor(int num_h, int num_d) : num_h_(num_h), num_d_(num_d){};
    const int num_h_;
    const int num_d_;
    __device__ thrust::tuple<Eigen::Vector3i, Voxel> operator()(
            size_t idx) const {
        int widx = idx / (num_h_ * num_d_);
        int hdidx = idx % (num_h_ * num_d_);
        int hidx = hdidx / num_d_;
        int didx = hdidx % num_d_;
        Eigen::Vector3i grid_index(widx, hidx, didx);
        return thrust::make_tuple(grid_index, geometry::Voxel(grid_index));
    }
};

struct add_voxel_color_functor {
    __device__ Voxel operator()(const Voxel &x, const Voxel &y) const {
        Voxel ans;
        ans.grid_index_ = x.grid_index_;
        ans.color_ = x.color_ + y.color_;
        return ans;
    }
};

struct devide_voxel_color_functor {
    __device__ Voxel operator()(const Voxel &x, int y) const {
        Voxel ans;
        ans.grid_index_ = x.grid_index_;
        ans.color_ = x.color_ / y;
        return ans;
    }
};

struct create_from_pointcloud_functor {
    create_from_pointcloud_functor(const Eigen::Vector3f &min_bound,
                                   float voxel_size,
                                   bool has_colors)
        : min_bound_(min_bound),
          voxel_size_(voxel_size),
          has_colors_(has_colors){};
    const Eigen::Vector3f min_bound_;
    const float voxel_size_;
    const bool has_colors_;
    __device__ thrust::tuple<Eigen::Vector3i, geometry::Voxel> operator()(
            const Eigen::Vector3f &point, const Eigen::Vector3f &color) const {
        Eigen::Vector3f ref_coord = (point - min_bound_) / voxel_size_;
        Eigen::Vector3i voxel_index;
        voxel_index << int(floor(ref_coord(0))), int(floor(ref_coord(1))),
                int(floor(ref_coord(2)));
        if (has_colors_) {
            return thrust::make_tuple(voxel_index,
                                      geometry::Voxel(voxel_index, color));
        } else {
            return thrust::make_tuple(voxel_index,
                                      geometry::Voxel(voxel_index));
        }
    }
};

struct create_from_trianglemesh_functor {
    create_from_trianglemesh_functor(const Eigen::Vector3f *vertices,
                                     const Eigen::Vector3i *triangles,
                                     int n_triangles,
                                     const Eigen::Vector3f &min_bound,
                                     float voxel_size,
                                     int num_h,
                                     int num_d)
        : vertices_(vertices),
          triangles_(triangles),
          n_triangles_(n_triangles),
          min_bound_(min_bound),
          voxel_size_(voxel_size),
          box_half_size_(Eigen::Vector3f(
                  voxel_size / 2, voxel_size / 2, voxel_size / 2)),
          num_h_(num_h),
          num_d_(num_d){};
    const Eigen::Vector3f *vertices_;
    const Eigen::Vector3i *triangles_;
    const int n_triangles_;
    const Eigen::Vector3f min_bound_;
    const float voxel_size_;
    const Eigen::Vector3f box_half_size_;
    const int num_h_;
    const int num_d_;
    __device__ thrust::tuple<Eigen::Vector3i, geometry::Voxel> operator()(
            size_t idx) const {
        int widx = idx / (num_h_ * num_d_);
        int hdidx = idx % (num_h_ * num_d_);
        int hidx = hdidx / num_d_;
        int didx = hdidx % num_d_;

        const Eigen::Vector3f box_center =
                min_bound_ + Eigen::Vector3f(widx, hidx, didx) * voxel_size_;
        for (int i = 0; i < n_triangles_; ++i) {
            Eigen::Vector3i tri = triangles_[i];
            const Eigen::Vector3f &v0 = vertices_[tri(0)];
            const Eigen::Vector3f &v1 = vertices_[tri(1)];
            const Eigen::Vector3f &v2 = vertices_[tri(2)];
            if (intersection_test::TriangleAABB(box_center, box_half_size_, v0,
                                                v1, v2)) {
                Eigen::Vector3i grid_index(widx, hidx, didx);
                return thrust::make_tuple(grid_index,
                                          geometry::Voxel(grid_index));
            }
        }
        return thrust::make_tuple(Eigen::Vector3i(-1, -1, -1),
                                  geometry::Voxel());
    }
};

}  // namespace

std::shared_ptr<VoxelGrid> VoxelGrid::CreateDense(const Eigen::Vector3f &origin,
                                                  float voxel_size,
                                                  float width,
                                                  float height,
                                                  float depth) {
    auto output = std::make_shared<VoxelGrid>();
    int num_w = int(std::round(width / voxel_size));
    int num_h = int(std::round(height / voxel_size));
    int num_d = int(std::round(depth / voxel_size));
    output->origin_ = origin;
    output->voxel_size_ = voxel_size;
    int n_total = num_w * num_h * num_d;
    output->voxels_keys_.resize(n_total);
    output->voxels_values_.resize(n_total);
    create_dense_functor func(num_h, num_d);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(n_total),
                      make_tuple_iterator(output->voxels_keys_.begin(),
                                          output->voxels_values_.begin()),
                      func);
    thrust::sort_by_key(output->voxels_keys_.begin(),
                        output->voxels_keys_.end(),
                        output->voxels_values_.begin());
    auto end = thrust::unique_by_key(output->voxels_keys_.begin(),
                                     output->voxels_keys_.end(),
                                     output->voxels_values_.begin());
    size_t out_size = thrust::distance(output->voxels_keys_.begin(), end.first);
    output->voxels_keys_.resize(out_size);
    output->voxels_values_.resize(out_size);
    return output;
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromPointCloudWithinBounds(
        const PointCloud &input,
        float voxel_size,
        const Eigen::Vector3f &min_bound,
        const Eigen::Vector3f &max_bound) {
    auto output = std::make_shared<VoxelGrid>();
    if (voxel_size <= 0.0) {
        utility::LogError("[VoxelGridFromPointCloud] voxel_size <= 0.");
    }

    if (voxel_size * std::numeric_limits<int>::max() <
        (max_bound - min_bound).maxCoeff()) {
        utility::LogError("[VoxelGridFromPointCloud] voxel_size is too small.");
    }
    output->voxel_size_ = voxel_size;
    output->origin_ = min_bound;
    utility::device_vector<Eigen::Vector3i> voxels_keys(input.points_.size());
    utility::device_vector<geometry::Voxel> voxels_values(input.points_.size());
    bool has_colors = input.HasColors();
    create_from_pointcloud_functor func(min_bound, voxel_size, has_colors);
    if (!has_colors) {
        thrust::transform(
                input.points_.begin(), input.points_.end(),
                thrust::make_constant_iterator(Eigen::Vector3f(0.0, 0.0, 0.0)),
                make_tuple_iterator(voxels_keys.begin(), voxels_values.begin()),
                func);
    } else {
        thrust::transform(
                input.points_.begin(), input.points_.end(),
                input.colors_.begin(),
                make_tuple_iterator(voxels_keys.begin(), voxels_values.begin()),
                func);
    }
    thrust::sort_by_key(voxels_keys.begin(), voxels_keys.end(),
                        voxels_values.begin());

    utility::device_vector<int> counts(voxels_keys.size());
    auto end = thrust::reduce_by_key(voxels_keys.begin(), voxels_keys.end(),
                                     thrust::make_constant_iterator(1),
                                     thrust::make_discard_iterator(),
                                     counts.begin());
    int n_out = thrust::distance(counts.begin(), end.second);
    counts.resize(n_out);
    output->voxels_keys_.resize(n_out);
    output->voxels_values_.resize(n_out);
    thrust::reduce_by_key(
            voxels_keys.begin(), voxels_keys.end(), voxels_values.begin(),
            output->voxels_keys_.begin(), output->voxels_values_.begin(),
            thrust::equal_to<Eigen::Vector3i>(), add_voxel_color_functor());
    thrust::transform(output->voxels_values_.begin(),
                      output->voxels_values_.end(), counts.begin(),
                      output->voxels_values_.begin(),
                      devide_voxel_color_functor());
    utility::LogDebug(
            "Pointcloud is voxelized from {:d} points to {:d} voxels.",
            (int)input.points_.size(), (int)output->voxels_keys_.size());
    return output;
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromPointCloud(
        const PointCloud &input, float voxel_size) {
    Eigen::Vector3f voxel_size3(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3f min_bound = input.GetMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3f max_bound = input.GetMaxBound() + voxel_size3 * 0.5;
    return CreateFromPointCloudWithinBounds(input, voxel_size, min_bound,
                                            max_bound);
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromTriangleMeshWithinBounds(
        const TriangleMesh &input,
        float voxel_size,
        const Eigen::Vector3f &min_bound,
        const Eigen::Vector3f &max_bound) {
    auto output = std::make_shared<VoxelGrid>();
    if (voxel_size <= 0.0) {
        utility::LogError("[CreateFromTriangleMesh] voxel_size <= 0.");
    }

    if (voxel_size * std::numeric_limits<int>::max() <
        (max_bound - min_bound).maxCoeff()) {
        utility::LogError("[CreateFromTriangleMesh] voxel_size is too small.");
    }
    output->voxel_size_ = voxel_size;
    output->origin_ = min_bound;

    Eigen::Vector3f grid_size = max_bound - min_bound;
    int num_w = int(std::round(grid_size(0) / voxel_size));
    int num_h = int(std::round(grid_size(1) / voxel_size));
    int num_d = int(std::round(grid_size(2) / voxel_size));
    size_t n_total = num_w * num_h * num_d;
    create_from_trianglemesh_functor func(
            thrust::raw_pointer_cast(input.vertices_.data()),
            thrust::raw_pointer_cast(input.triangles_.data()),
            input.triangles_.size(), min_bound, voxel_size, num_h, num_d);
    output->voxels_keys_.resize(n_total);
    output->voxels_values_.resize(n_total);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_total),
                      make_tuple_iterator(output->voxels_keys_.begin(),
                                          output->voxels_values_.begin()),
                      func);
    auto begin = make_tuple_iterator(output->voxels_keys_.begin(),
                                     output->voxels_values_.begin());
    auto end = thrust::remove_if(
            begin,
            make_tuple_iterator(output->voxels_keys_.end(),
                                output->voxels_values_.end()),
            [] __device__(
                    const thrust::tuple<Eigen::Vector3i, geometry::Voxel> &x)
                    -> bool {
                Eigen::Vector3i idxs = thrust::get<0>(x);
                return idxs[0] < 0;
            });
    size_t n_out = thrust::distance(begin, end);
    output->voxels_keys_.resize(n_out);
    output->voxels_values_.resize(n_out);
    return output;
}

std::shared_ptr<VoxelGrid> VoxelGrid::CreateFromTriangleMesh(
        const TriangleMesh &input, float voxel_size) {
    Eigen::Vector3f voxel_size3(voxel_size, voxel_size, voxel_size);
    Eigen::Vector3f min_bound = input.GetMinBound() - voxel_size3 * 0.5;
    Eigen::Vector3f max_bound = input.GetMaxBound() + voxel_size3 * 0.5;
    return CreateFromTriangleMeshWithinBounds(input, voxel_size, min_bound,
                                              max_bound);
}