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
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>

#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/geometry_utils.h"
#include "cupoch/geometry/image.h"
#include "cupoch/knn/kdtree_flann.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/helper.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace geometry {

namespace {

template <class... Args>
struct check_nan_functor {
    check_nan_functor(bool remove_nan, bool remove_infinite)
        : remove_nan_(remove_nan), remove_infinite_(remove_infinite){};
    const bool remove_nan_;
    const bool remove_infinite_;
    __device__ bool operator()(const thrust::tuple<Args...> &x) const {
        const Eigen::Vector3f &point = thrust::get<0>(x);
        bool is_nan = remove_nan_ &&
                      (isnan(point(0)) || isnan(point(1)) || isnan(point(2)));
        bool is_infinite =
                remove_infinite_ &&
                (isinf(point(0)) || isinf(point(1)) || isinf(point(2)));
        return is_nan || is_infinite;
    }
};

struct gaussian_filter_functor {
    gaussian_filter_functor(const Eigen::Vector3f *points,
                            const Eigen::Vector3f *normals,
                            const Eigen::Vector3f *colors,
                            const int *indices,
                            const float *dists,
                            float sigma2,
                            size_t num_max_search_points,
                            bool has_normal,
                            bool has_color)
        : points_(points),
          normals_(normals),
          colors_(colors),
          indices_(indices),
          dists_(dists),
          sigma2_(sigma2),
          num_max_search_points_(num_max_search_points),
          has_normal_(has_normal),
          has_color_(has_color){};
    const Eigen::Vector3f *points_;
    const Eigen::Vector3f *normals_;
    const Eigen::Vector3f *colors_;
    const int *indices_;
    const float *dists_;
    const float sigma2_;
    const size_t num_max_search_points_;
    const bool has_normal_;
    const bool has_color_;
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f>
    operator()(size_t idx) const {
        float total_weight = 0.0;
        Eigen::Vector3f res_p = Eigen::Vector3f::Zero();
        Eigen::Vector3f res_n = Eigen::Vector3f::Zero();
        Eigen::Vector3f res_c = Eigen::Vector3f::Zero();
        for (int i = 0; i < num_max_search_points_; ++i) {
            const int j = idx * num_max_search_points_ + i;
            const int idx_j = __ldg(&indices_[j]);
            if (idx_j >= 0) {
                float weight = exp(-0.5 * dists_[j] / sigma2_);
                res_p += weight * points_[idx_j];
                if (has_normal_) res_n += weight * normals_[idx_j];
                if (has_color_) res_c += weight * colors_[idx_j];
                total_weight += weight;
            }
        }
        res_p /= total_weight;
        res_n /= total_weight;
        res_c /= total_weight;
        return thrust::make_tuple(res_p, res_n, res_c);
    }
};

template <class... Args>
struct pass_through_filter_functor {
    pass_through_filter_functor(int axis_no, float min_bound, float max_bound)
        : axis_no_(axis_no), min_bound_(min_bound), max_bound_(max_bound){};
    const int axis_no_;
    const float min_bound_;
    const float max_bound_;
    __device__ bool operator()(
            const thrust::tuple<Eigen::Vector3f, Args...> &x) const {
        float val = thrust::get<0>(x)[axis_no_];
        return val < min_bound_ || max_bound_ < val;
    }
};

struct compute_farthest_index_functor {
    compute_farthest_index_functor(
        const Eigen::Vector3f *points,
        float *distances,
        size_t farthest_index)
        : points_(points),
          distances_(distances),
          farthest_index_(farthest_index){};
    const Eigen::Vector3f *points_;
    float *distances_;
    size_t farthest_index_;
    __device__ thrust::tuple<size_t, float> operator()(size_t idx) const {
        const Eigen::Vector3f &selected = points_[farthest_index_];
        float dist = (points_[idx] - selected).squaredNorm();
        distances_[idx] = min(distances_[idx], dist);
        return thrust::make_tuple(idx, distances_[idx]);
    }
};

}  // namespace

PointCloud::PointCloud() : GeometryBase3D(Geometry::GeometryType::PointCloud) {}
PointCloud::PointCloud(const thrust::host_vector<Eigen::Vector3f> &points)
    : GeometryBase3D(Geometry::GeometryType::PointCloud), points_(points) {}
PointCloud::PointCloud(const std::vector<Eigen::Vector3f> &points)
    : GeometryBase3D(Geometry::GeometryType::PointCloud), points_(points) {}
PointCloud::PointCloud(const utility::device_vector<Eigen::Vector3f> &points)
    : GeometryBase3D(Geometry::GeometryType::PointCloud), points_(points) {}
PointCloud::PointCloud(const PointCloud &other)
    : GeometryBase3D(Geometry::GeometryType::PointCloud),
      points_(other.points_),
      normals_(other.normals_),
      colors_(other.colors_),
      covariances_(other.covariances_) {}

PointCloud::~PointCloud() {}

PointCloud &PointCloud::operator=(const PointCloud &other) {
    points_ = other.points_;
    normals_ = other.normals_;
    colors_ = other.colors_;
    covariances_ = other.covariances_;
    return *this;
}

void PointCloud::SetPoints(const thrust::host_vector<Eigen::Vector3f> &points) {
    points_ = points;
}

void PointCloud::SetPoints(const std::vector<Eigen::Vector3f> &points) {
    points_.resize(points.size());
    copy_host_to_device(points, points_);
}

std::vector<Eigen::Vector3f> PointCloud::GetPoints() const {
    std::vector<Eigen::Vector3f> points(points_.size());
    copy_device_to_host(points_, points);
    return points;
}

void PointCloud::SetNormals(
        const thrust::host_vector<Eigen::Vector3f> &normals) {
    normals_ = normals;
}

void PointCloud::SetNormals(const std::vector<Eigen::Vector3f> &normals) {
    normals_.resize(normals.size());
    copy_host_to_device(normals, normals_);
}

std::vector<Eigen::Vector3f> PointCloud::GetNormals() const {
    std::vector<Eigen::Vector3f> normals(normals_.size());
    copy_device_to_host(normals_, normals);
    return normals;
}

void PointCloud::SetColors(const thrust::host_vector<Eigen::Vector3f> &colors) {
    colors_ = colors;
}

void PointCloud::SetColors(const std::vector<Eigen::Vector3f> &colors) {
    colors_.resize(colors.size());
    copy_host_to_device(colors, colors_);
}

std::vector<Eigen::Vector3f> PointCloud::GetColors() const {
    std::vector<Eigen::Vector3f> colors(colors_.size());
    copy_device_to_host(colors_, colors);
    return colors;
}

PointCloud &PointCloud::Clear() {
    points_.clear();
    normals_.clear();
    colors_.clear();
    covariances_.clear();
    return *this;
}

bool PointCloud::IsEmpty() const { return !HasPoints(); }

Eigen::Vector3f PointCloud::GetMinBound() const {
    return utility::ComputeMinBound<3>(points_);
}

Eigen::Vector3f PointCloud::GetMaxBound() const {
    return utility::ComputeMaxBound<3>(points_);
}

Eigen::Vector3f PointCloud::GetCenter() const {
    return utility::ComputeCenter<3>(points_);
}

AxisAlignedBoundingBox<3> PointCloud::GetAxisAlignedBoundingBox() const {
    return AxisAlignedBoundingBox<3>::CreateFromPoints(points_);
}

OrientedBoundingBox PointCloud::GetOrientedBoundingBox() const {
    return OrientedBoundingBox::CreateFromPoints(points_);
}

PointCloud &PointCloud::Translate(const Eigen::Vector3f &translation,
                                  bool relative) {
    TranslatePoints<3>(translation, points_, relative);
    return *this;
}

PointCloud &PointCloud::Scale(const float scale, bool center) {
    ScalePoints<3>(scale, points_, center);
    return *this;
}

PointCloud &PointCloud::Rotate(const Eigen::Matrix3f &R, bool center) {
    RotatePoints<3>(utility::GetStream(0), R, points_, center);
    RotateNormals(utility::GetStream(1), R, normals_);
    RotateCovariances(utility::GetStream(2), R, covariances_);
    cudaSafeCall(cudaDeviceSynchronize());
    return *this;
}

PointCloud &PointCloud::operator+=(const PointCloud &cloud) {
    // We do not use std::vector::insert to combine std::vector because it will
    // crash if the pointcloud is added to itself.
    if (cloud.IsEmpty()) return (*this);
    size_t old_vert_num = points_.size();
    size_t add_vert_num = cloud.points_.size();
    size_t new_vert_num = old_vert_num + add_vert_num;
    if ((!HasPoints() || HasNormals()) && cloud.HasNormals()) {
        normals_.resize(new_vert_num);
        thrust::copy(cloud.normals_.begin(), cloud.normals_.end(),
                     normals_.begin() + old_vert_num);
    } else {
        normals_.clear();
    }
    if ((!HasPoints() || HasColors()) && cloud.HasColors()) {
        colors_.resize(new_vert_num);
        thrust::copy(cloud.colors_.begin(), cloud.colors_.end(),
                     colors_.begin() + old_vert_num);
    } else {
        colors_.clear();
    }
    if ((!HasPoints() || HasCovariances()) && cloud.HasCovariances()) {
        covariances_.resize(new_vert_num);
        thrust::copy(cloud.covariances_.begin(), cloud.covariances_.end(),
                     covariances_.begin() + old_vert_num);
    } else {
        covariances_.clear();
    }
    points_.resize(new_vert_num);
    thrust::copy(cloud.points_.begin(), cloud.points_.end(),
                 points_.begin() + old_vert_num);
    return (*this);
}

PointCloud PointCloud::operator+(const PointCloud &cloud) const {
    return (PointCloud(*this) += cloud);
}

PointCloud &PointCloud::NormalizeNormals() {
    thrust::for_each(normals_.begin(), normals_.end(),
                     [] __device__(Eigen::Vector3f & nl) { nl.normalize(); });
    return *this;
}

PointCloud &PointCloud::PaintUniformColor(const Eigen::Vector3f &color) {
    ResizeAndPaintUniformColor(colors_, points_.size(), color);
    return *this;
}

PointCloud &PointCloud::Transform(const Eigen::Matrix4f &transformation) {
    TransformPoints<3>(utility::GetStream(0), transformation, points_);
    TransformNormals(utility::GetStream(1), transformation, normals_);
    TransformCovariances(utility::GetStream(2), transformation, covariances_);
    cudaSafeCall(cudaDeviceSynchronize());
    return *this;
}

std::shared_ptr<PointCloud> PointCloud::FarthestPointDownSample(
        size_t num_samples) const {
    if (num_samples == 0) {
        return std::make_shared<PointCloud>();
    } else if (num_samples == points_.size()) {
        return std::make_shared<PointCloud>(*this);
    } else if (num_samples > points_.size()) {
        utility::LogError(
                "Illegal number of samples: {}, must <= point size: {}",
                num_samples, points_.size());
    }
    // We can also keep track of the non-selected indices with unordered_set,
    // but since typically num_samples << num_points, it may not be worth it.
    std::vector<size_t> selected_indices;
    selected_indices.reserve(num_samples);
    const size_t num_points = points_.size();
    utility::device_vector<float> distances(
        num_points, std::numeric_limits<float>::infinity());
    size_t farthest_index = 0;
    for (size_t i = 0; i < num_samples; i++) {
        selected_indices.push_back(farthest_index);
        compute_farthest_index_functor func(
            thrust::raw_pointer_cast(points_.data()),
            thrust::raw_pointer_cast(distances.data()),
            farthest_index);
        auto res = thrust::transform_reduce(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_points),
            func,
            thrust::make_tuple<size_t, float>(static_cast<size_t>(farthest_index), 0.0f),
            [] __host__ __device__(const thrust::tuple<size_t, float> &a,
                              const thrust::tuple<size_t, float> &b) -> thrust::tuple<size_t, float> {
                return thrust::get<1>(a) > thrust::get<1>(b) ? a : b;
            });
        farthest_index = thrust::get<0>(res);
    }
    return SelectByIndex(selected_indices);
}

std::shared_ptr<PointCloud> PointCloud::Crop(
        const AxisAlignedBoundingBox<3> &bbox) const {
    if (bbox.IsEmpty()) {
        utility::LogError(
                "[CropPointCloud] AxisAlignedBoundingBox either has zeros "
                "size, or has wrong bounds.");
    }
    return SelectByIndex(bbox.GetPointIndicesWithinBoundingBox(points_));
}

std::shared_ptr<PointCloud> PointCloud::Crop(
        const OrientedBoundingBox &bbox) const {
    if (bbox.IsEmpty()) {
        utility::LogError(
                "[CropPointCloud] AxisAlignedBoundingBox either has zeros "
                "size, or has wrong bounds.");
    }
    return SelectByIndex(bbox.GetPointIndicesWithinBoundingBox(points_));
}

PointCloud &PointCloud::RemoveNoneFinitePoints(bool remove_nan,
                                               bool remove_infinite) {
    bool has_normal = HasNormals();
    bool has_color = HasColors();
    size_t old_point_num = points_.size();
    size_t k = 0;
    auto runs = [=] (auto&... params) {
        remove_if_vectors(
                utility::exec_policy(0),
                check_nan_functor<typename std::remove_reference_t<decltype(params)>::value_type...>(remove_nan, remove_infinite),
                params...);
    };
    if (!has_normal && !has_color) {
        runs(points_);
    } else if (has_normal && !has_color) {
        runs(points_, normals_);
    } else if (!has_normal && has_color) {
        runs(points_, colors_);
    } else {
        runs(points_, normals_, colors_);
    }
    utility::LogDebug(
            "[RemoveNoneFinitePoints] {:d} nan points have been removed.",
            (int)(old_point_num - k));
    return *this;
}

std::shared_ptr<PointCloud> PointCloud::GaussianFilter(
        float search_radius, float sigma2, size_t num_max_search_points) {
    auto out = std::make_shared<PointCloud>();
    if (search_radius <= 0 || sigma2 <= 0 || num_max_search_points <= 0) {
        utility::LogError(
                "[GaussianFilter] Illegal input parameters, radius and sigma2 "
                "must be positive.");
        return out;
    }
    bool has_normal = HasNormals();
    bool has_color = HasColors();
    knn::KDTreeFlann kdtree;
    kdtree.SetRawData(ConvertVector3fVectorRef(*this));
    utility::device_vector<int> indices;
    utility::device_vector<float> dist;
    kdtree.SearchRadius(points_, search_radius, num_max_search_points, indices,
                        dist);
    size_t n_pt = points_.size();
    out->points_.resize(n_pt);
    if (has_normal) out->normals_.resize(n_pt);
    if (has_color) out->colors_.resize(n_pt);
    gaussian_filter_functor func(thrust::raw_pointer_cast(points_.data()),
                                 thrust::raw_pointer_cast(normals_.data()),
                                 thrust::raw_pointer_cast(colors_.data()),
                                 thrust::raw_pointer_cast(indices.data()),
                                 thrust::raw_pointer_cast(dist.data()), sigma2,
                                 num_max_search_points, has_normal, has_color);
    auto runs = [size = points_.size(), &func] (auto&&... params) {
        thrust::transform(
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator(size),
                make_tuple_iterator(params...), func);
    };
    if (has_normal && has_color) {
        runs(out->points_.begin(), out->normals_.begin(), out->colors_.begin());
    } else if (has_normal) {
        runs(out->points_.begin(), out->normals_.begin(),
             thrust::make_discard_iterator());
    } else if (has_color) {
        runs(out->points_.begin(), thrust::make_discard_iterator(),
             out->colors_.begin());
    } else {
        runs(out->points_.begin(),
             thrust::make_discard_iterator(),
             thrust::make_discard_iterator());
    }
    return out;
}

std::shared_ptr<PointCloud> PointCloud::PassThroughFilter(size_t axis_no,
                                                          float min_bound,
                                                          float max_bound) {
    auto out = std::make_shared<PointCloud>();
    if (axis_no >= 3) {
        utility::LogError(
                "[PassThroughFilter] Illegal input parameters, axis_no "
                "must be 0, 1 or 2.");
        return out;
    }
    *out = *this;
    bool has_normal = HasNormals();
    bool has_color = HasColors();
    auto runs = [=, &points = out->points_] (auto&... params) {
        remove_if_vectors(
                utility::exec_policy(0),
                pass_through_filter_functor<typename std::remove_reference_t<decltype(params)>::value_type...>(
                        axis_no, min_bound, max_bound),
                points, params...);
    };
    if (has_normal && has_color) {
        runs(out->normals_, out->colors_);
    } else if (has_normal) {
        runs(out->normals_);
    } else if (has_color) {
        runs(out->colors_);
    } else {
        runs();
    }
    return out;
}

}  // namespace geometry
}  // namespace cupoch