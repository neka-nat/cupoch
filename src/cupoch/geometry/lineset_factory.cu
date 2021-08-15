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
#include <Eigen/Dense>

#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/lineset.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/trianglemesh.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/platform.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct convert_trianglemesh_line_functor {
    convert_trianglemesh_line_functor(const Eigen::Vector3i *triangles,
                                      Eigen::Vector2i *lines)
        : triangles_(triangles), lines_(lines){};
    const Eigen::Vector3i *triangles_;
    Eigen::Vector2i *lines_;
    __device__ void operator()(size_t idx) const {
        const Eigen::Vector3i &vidx = triangles_[idx];
        thrust::minimum<int> min;
        thrust::maximum<int> max;
        lines_[3 * idx] =
                Eigen::Vector2i(min(vidx[0], vidx[1]), max(vidx[0], vidx[1]));
        lines_[3 * idx + 1] =
                Eigen::Vector2i(min(vidx[1], vidx[2]), max(vidx[1], vidx[2]));
        lines_[3 * idx + 2] =
                Eigen::Vector2i(min(vidx[2], vidx[0]), max(vidx[2], vidx[0]));
    }
};

}  // namespace

template <>
template <>
std::shared_ptr<LineSet<3>> LineSet<3>::CreateFromPointCloudCorrespondences(
        const PointCloud &cloud0,
        const PointCloud &cloud1,
        const utility::device_vector<thrust::pair<int, int>> &correspondences) {
    auto lineset_ptr = std::make_shared<LineSet<3>>();
    const size_t point0_size = cloud0.points_.size();
    const size_t point1_size = cloud1.points_.size();
    const size_t corr_size = correspondences.size();
    lineset_ptr->points_.resize(point0_size + point1_size);
    lineset_ptr->lines_.resize(corr_size);
    thrust::copy_n(utility::exec_policy(utility::GetStream(0))
                           ->on(utility::GetStream(0)),
                   cloud0.points_.begin(), point0_size,
                   lineset_ptr->points_.begin());
    thrust::copy_n(utility::exec_policy(utility::GetStream(1))
                           ->on(utility::GetStream(1)),
                   cloud1.points_.begin(), point1_size,
                   lineset_ptr->points_.begin() + point0_size);
    thrust::transform(utility::exec_policy(utility::GetStream(2))
                              ->on(utility::GetStream(2)),
                      correspondences.begin(), correspondences.end(),
                      lineset_ptr->lines_.begin(),
                      [=] __device__(const thrust::pair<int, int> &corrs) {
                          return Eigen::Vector2i(corrs.first,
                                                 point0_size + corrs.second);
                      });
    cudaSafeCall(cudaDeviceSynchronize());
    return lineset_ptr;
}

template <>
template <>
std::shared_ptr<LineSet<2>> LineSet<2>::CreateFromPointCloudCorrespondences(
        const PointCloud &cloud0,
        const PointCloud &cloud1,
        const utility::device_vector<thrust::pair<int, int>> &correspondences) {
    utility::LogError(
            "LineSet<2>::CreateFromPointCloudCorrespondences is not supported");
    return std::make_shared<LineSet<2>>();
}

template <>
template <>
std::shared_ptr<LineSet<3>> LineSet<3>::CreateFromTriangleMesh(
        const TriangleMesh &mesh) {
    auto lineset_ptr = std::make_shared<LineSet<3>>();
    lineset_ptr->points_.resize(mesh.vertices_.size());
    lineset_ptr->lines_.resize(mesh.triangles_.size() * 3);
    convert_trianglemesh_line_functor func(
            thrust::raw_pointer_cast(mesh.triangles_.data()),
            thrust::raw_pointer_cast(lineset_ptr->lines_.data()));
    thrust::copy(utility::exec_policy(utility::GetStream(0))
                         ->on(utility::GetStream(0)),
                 mesh.vertices_.begin(), mesh.vertices_.end(),
                 lineset_ptr->points_.begin());
    thrust::for_each(utility::exec_policy(utility::GetStream(1))
                             ->on(utility::GetStream(1)),
                     thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(mesh.triangles_.size()),
                     func);
    auto end = thrust::unique(utility::exec_policy(utility::GetStream(1))
                                      ->on(utility::GetStream(1)),
                              lineset_ptr->lines_.begin(),
                              lineset_ptr->lines_.end());
    lineset_ptr->lines_.resize(
            thrust::distance(lineset_ptr->lines_.begin(), end));
    cudaSafeCall(cudaDeviceSynchronize());
    return lineset_ptr;
}

template <>
template <>
std::shared_ptr<LineSet<2>> LineSet<2>::CreateFromTriangleMesh(
        const TriangleMesh &mesh) {
    utility::LogError("LineSet<2>::CreateFromTriangleMesh is not supported");
    return std::make_shared<LineSet<2>>();
}

template <>
template <>
std::shared_ptr<LineSet<3>> LineSet<3>::CreateFromOrientedBoundingBox(
        const OrientedBoundingBox &box) {
    auto line_set = std::make_shared<LineSet<3>>();
    const auto points = box.GetBoxPoints();
    for (const auto &pt : points) line_set->points_.push_back(pt);
    line_set->lines_.push_back(Eigen::Vector2i(0, 1));
    line_set->lines_.push_back(Eigen::Vector2i(1, 7));
    line_set->lines_.push_back(Eigen::Vector2i(7, 2));
    line_set->lines_.push_back(Eigen::Vector2i(2, 0));
    line_set->lines_.push_back(Eigen::Vector2i(3, 6));
    line_set->lines_.push_back(Eigen::Vector2i(6, 4));
    line_set->lines_.push_back(Eigen::Vector2i(4, 5));
    line_set->lines_.push_back(Eigen::Vector2i(5, 3));
    line_set->lines_.push_back(Eigen::Vector2i(0, 3));
    line_set->lines_.push_back(Eigen::Vector2i(1, 6));
    line_set->lines_.push_back(Eigen::Vector2i(7, 4));
    line_set->lines_.push_back(Eigen::Vector2i(2, 5));
    line_set->PaintUniformColor(box.color_);
    return line_set;
}

template <>
template <>
std::shared_ptr<LineSet<2>> LineSet<2>::CreateFromOrientedBoundingBox(
        const OrientedBoundingBox &box) {
    utility::LogError(
            "LineSet<2>::CreateFromOrientedBoundingBox is not supported");
    return std::make_shared<LineSet<2>>();
}

template <>
template <>
std::shared_ptr<LineSet<3>> LineSet<3>::CreateFromAxisAlignedBoundingBox(
        const AxisAlignedBoundingBox<3> &box) {
    auto line_set = std::make_shared<LineSet<3>>();
    const auto points = box.GetBoxPoints();
    for (const auto &pt : points) line_set->points_.push_back(pt);
    line_set->lines_.push_back(Eigen::Vector2i(0, 1));
    line_set->lines_.push_back(Eigen::Vector2i(1, 7));
    line_set->lines_.push_back(Eigen::Vector2i(7, 2));
    line_set->lines_.push_back(Eigen::Vector2i(2, 0));
    line_set->lines_.push_back(Eigen::Vector2i(3, 6));
    line_set->lines_.push_back(Eigen::Vector2i(6, 4));
    line_set->lines_.push_back(Eigen::Vector2i(4, 5));
    line_set->lines_.push_back(Eigen::Vector2i(5, 3));
    line_set->lines_.push_back(Eigen::Vector2i(0, 3));
    line_set->lines_.push_back(Eigen::Vector2i(1, 6));
    line_set->lines_.push_back(Eigen::Vector2i(7, 4));
    line_set->lines_.push_back(Eigen::Vector2i(2, 5));
    line_set->PaintUniformColor(box.color_);
    return line_set;
}

template <>
template <>
std::shared_ptr<LineSet<2>> LineSet<2>::CreateFromAxisAlignedBoundingBox(
        const AxisAlignedBoundingBox<2> &box) {
    utility::LogError(
            "LineSet<2>::CreateFromAxisAlignedBoundingBox is not supported");
    return std::make_shared<LineSet<2>>();
}


template <>
template <>
std::shared_ptr<geometry::LineSet<3>> LineSet<3>::CreateCameraMarker(
        const camera::PinholeCameraIntrinsic& intrinsic,
        const Eigen::Matrix4f& extrinsic,
        float marker_size) {
    thrust::host_vector<Eigen::Vector3f> points(5);
    thrust::host_vector<Eigen::Vector2i> lines(8);
    const auto focal_length = intrinsic.GetFocalLength();
    const auto x = intrinsic.width_ / focal_length.first * marker_size;
    const auto y = intrinsic.height_ / focal_length.second * marker_size;
    const Eigen::Vector3f local_pt1(x, y, marker_size);
    const Eigen::Vector3f local_pt2(x, -y, marker_size);
    const Eigen::Vector3f local_pt3(-x, -y, marker_size);
    const Eigen::Vector3f local_pt4(-x, y, marker_size);
    const Eigen::Matrix4f inv_ext = utility::InverseTransform(extrinsic);
    points[0] = inv_ext.block<3, 1>(0, 3);
    points[1] = inv_ext.block<3, 3>(0, 0) * local_pt1 + inv_ext.block<3, 1>(0, 3);
    points[2] = inv_ext.block<3, 3>(0, 0) * local_pt2 + inv_ext.block<3, 1>(0, 3);
    points[3] = inv_ext.block<3, 3>(0, 0) * local_pt3 + inv_ext.block<3, 1>(0, 3);
    points[4] = inv_ext.block<3, 3>(0, 0) * local_pt4 + inv_ext.block<3, 1>(0, 3);
    lines[0] = Eigen::Vector2i(0, 1);
    lines[1] = Eigen::Vector2i(0, 2);
    lines[2] = Eigen::Vector2i(0, 3);
    lines[3] = Eigen::Vector2i(0, 4);
    lines[4] = Eigen::Vector2i(1, 2);
    lines[5] = Eigen::Vector2i(2, 3);
    lines[6] = Eigen::Vector2i(3, 4);
    lines[7] = Eigen::Vector2i(4, 1);
    auto out = std::make_shared<LineSet<3>>(points, lines);
    return out;
}

template <>
template <>
std::shared_ptr<geometry::LineSet<2>> LineSet<2>::CreateCameraMarker(
        const camera::PinholeCameraIntrinsic& intrinsic,
        const Eigen::Matrix4f& extrinsic,
        float marker_size) {
    utility::LogError(
            "LineSet<2>::CreateCameraMarker is not supported");
    return std::make_shared<LineSet<2>>();
}
