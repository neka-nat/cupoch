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
#include <thrust/host_vector.h>

#include "cupoch/geometry/geometry_base.h"
#include "cupoch/knn/kdtree_search_param.h"
#include "cupoch/utility/device_vector.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {

namespace camera {
class PinholeCameraIntrinsic;
}

namespace geometry {

class Image;
class RGBDImage;
class LaserScanBuffer;
class OccupancyGrid;
class OrientedBoundingBox;

class PointCloud : public GeometryBase3D {
public:
    PointCloud();
    PointCloud(const thrust::host_vector<Eigen::Vector3f> &points);
    PointCloud(const std::vector<Eigen::Vector3f> &points);
    PointCloud(const utility::device_vector<Eigen::Vector3f> &points);
    PointCloud(const PointCloud &other);
    ~PointCloud();
    PointCloud &operator=(const PointCloud &other);

    void SetPoints(const thrust::host_vector<Eigen::Vector3f> &points);
    void SetPoints(const std::vector<Eigen::Vector3f> &points);
    std::vector<Eigen::Vector3f> GetPoints() const;

    void SetNormals(const thrust::host_vector<Eigen::Vector3f> &normals);
    void SetNormals(const std::vector<Eigen::Vector3f> &normals);
    std::vector<Eigen::Vector3f> GetNormals() const;

    void SetColors(const thrust::host_vector<Eigen::Vector3f> &colors);
    void SetColors(const std::vector<Eigen::Vector3f> &colors);
    std::vector<Eigen::Vector3f> GetColors() const;

    PointCloud &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3f GetMinBound() const override;
    Eigen::Vector3f GetMaxBound() const override;
    Eigen::Vector3f GetCenter() const override;
    AxisAlignedBoundingBox<3> GetAxisAlignedBoundingBox() const override;
    OrientedBoundingBox GetOrientedBoundingBox() const;
    PointCloud &Transform(const Eigen::Matrix4f &transformation) override;
    PointCloud &Translate(const Eigen::Vector3f &translation,
                          bool relative = true) override;
    PointCloud &Scale(const float scale, bool center = true) override;
    PointCloud &Rotate(const Eigen::Matrix3f &R, bool center = true) override;

    PointCloud &operator+=(const PointCloud &cloud);
    PointCloud operator+(const PointCloud &cloud) const;

    /// Returns 'true' if the point cloud contains points.
    __host__ __device__ bool HasPoints() const { return !points_.empty(); }

    /// Returns `true` if the point cloud contains point normals.
    __host__ __device__ bool HasNormals() const {
        return !points_.empty() && normals_.size() == points_.size();
    }

    /// Returns `true` if the point cloud contains point colors.
    __host__ __device__ bool HasColors() const {
        return !points_.empty() && colors_.size() == points_.size();
    }

    /// Returns 'true' if the point cloud contains per-point covariance matrix.
    __host__ __device__ bool HasCovariances() const {
        return !points_.empty() && covariances_.size() == points_.size();
    }

    /// Normalize point normals to length 1.
    PointCloud &NormalizeNormals();

    /// Assigns each point in the PointCloud the same color \param color.
    PointCloud &PaintUniformColor(const Eigen::Vector3f &color);

    /// \brief Remove all points fromt he point cloud that have a nan entry, or
    /// infinite entries.
    ///
    /// Also removes the corresponding normals and color entries.
    ///
    /// \param remove_nan Remove NaN values from the PointCloud.
    /// \param remove_infinite Remove infinite values from the PointCloud.
    PointCloud &RemoveNoneFinitePoints(bool remove_nan = true,
                                       bool remove_infinite = true);

    /// \brief Function to select points from \p input pointcloud into
    /// \p output pointcloud.
    ///
    /// Points with indices in \param indices are selected.
    ///
    /// \param indices Indices of points to be selected.
    /// \param invert Set to `True` to invert the selection of indices.
    std::shared_ptr<PointCloud> SelectByIndex(
            const utility::device_vector<size_t> &indices,
            bool invert = false) const;

    std::shared_ptr<PointCloud> SelectByMask(
            const utility::device_vector<bool> &mask,
            bool invert = false) const;

    std::shared_ptr<PointCloud> SelectByIndex(
            const std::vector<size_t> &indices,
            bool invert = false) const;

    std::shared_ptr<PointCloud> SelectByMask(
            const std::vector<bool> &mask,
            bool invert = false) const;

    /// Function to downsample \param input pointcloud into output pointcloud
    /// with a voxel \param voxel_size defines the resolution of the voxel grid,
    /// smaller value leads to denser output point cloud. Normals and colors are
    /// averaged if they exist.
    std::shared_ptr<PointCloud> VoxelDownSample(float voxel_size) const;

    /// Function to downsample \param input pointcloud into output pointcloud
    /// uniformly \param every_k_points indicates the sample rate.
    std::shared_ptr<PointCloud> UniformDownSample(size_t every_k_points) const;

    std::tuple<std::shared_ptr<PointCloud>, utility::device_vector<size_t>>
    RemoveRadiusOutliers(size_t nb_points, float search_radius) const;

    std::tuple<std::shared_ptr<PointCloud>, utility::device_vector<size_t>>
    RemoveStatisticalOutliers(size_t nb_neighbors, float std_ratio) const;

    std::shared_ptr<PointCloud> GaussianFilter(float search_radius,
                                               float sigma2,
                                               size_t num_max_search_points = 50);

    std::shared_ptr<PointCloud> PassThroughFilter(size_t axis_no,
                                                  float min_bound,
                                                  float max_bound);

    /// \brief Function to downsample input pointcloud into output pointcloud
    /// with a set of points has farthest distance.
    ///
    /// The sample is performed by selecting the farthest point from previous
    /// selected points iteratively.
    ///
    /// \param num_samples Number of points to be sampled.
    std::shared_ptr<PointCloud> FarthestPointDownSample(
            size_t num_samples) const;

    /// Function to crop pointcloud into output pointcloud
    /// All points with coordinates outside the bounding box \param bbox are
    /// clipped.
    std::shared_ptr<PointCloud> Crop(
            const AxisAlignedBoundingBox<3> &bbox) const;

    /// \brief Function to crop pointcloud into output pointcloud
    ///
    /// All points with coordinates outside the bounding box \p bbox are
    /// clipped.
    ///
    /// \param bbox OrientedBoundingBox to crop points.
    std::shared_ptr<PointCloud> Crop(const OrientedBoundingBox &bbox) const;

    /// Function to compute the normals of a point cloud
    /// \param cloud is the input point cloud. It also stores the output
    /// normals. Normals are oriented with respect to the input point cloud if
    /// normals exist in the input. \param search_param The KDTree search
    /// parameters
    bool EstimateNormals(
            const knn::KDTreeSearchParam &search_param = knn::KDTreeSearchParamKNN());

    /// Function to orient the normals of a point cloud
    /// \param cloud is the input point cloud. It must have normals.
    /// Normals are oriented with respect to \param orientation_reference
    bool OrientNormalsToAlignWithDirection(
            const Eigen::Vector3f &orientation_reference =
                    Eigen::Vector3f(0.0, 0.0, 1.0));

    /// Cluster PointCloud using the DBSCAN algorithm
    /// Ester et al., "A Density-Based Algorithm for Discovering Clusters
    /// in Large Spatial Databases with Noise", 1996
    /// Returns a vector of point labels, -1 indicates noise according to
    /// the algorithm.
    std::unique_ptr<utility::device_vector<int>> ClusterDBSCAN(
            float eps,
            size_t min_points,
            bool print_progress = false,
            size_t max_edges = knn::NUM_MAX_NN) const;

    /// \brief Segment PointCloud plane using the RANSAC algorithm.
    ///
    /// \param distance_threshold Max distance a point can be from the plane
    /// model, and still be considered an inlier.
    /// \param ransac_n Number of initial points to be considered inliers in
    /// each iteration.
    /// \param num_iterations Number of iterations.
    /// \return Returns the plane model ax + by + cz + d = 0 and the indices of
    /// the plane inliers.
    std::tuple<Eigen::Vector4f, utility::device_vector<size_t>> SegmentPlane(
            float distance_threshold = 0.01,
            size_t ransac_n = 3,
            size_t num_iterations = 100) const;

    /// Factory function to create a pointcloud from a depth image and a camera
    /// model (PointCloudFactory.cpp)
    /// The input depth image can be either a float image, or a uint16_t image.
    /// In the latter case, the depth is scaled by 1 / depth_scale, and
    /// truncated at depth_trunc distance. The depth image is also sampled with
    /// stride, in order to support (fast) coarse point cloud extraction. Return
    /// an empty pointcloud if the conversion fails.
    static std::shared_ptr<PointCloud> CreateFromDepthImage(
            const Image &depth,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4f &extrinsic = Eigen::Matrix4f::Identity(),
            float depth_scale = 1000.0,
            float depth_trunc = 1000.0,
            int stride = 1);

    /// Factory function to create a pointcloud from an RGB-D image and a camera
    /// model (PointCloudFactory.cpp)
    /// Return an empty pointcloud if the conversion fails.
    /// If \param project_valid_depth_only is true, return point cloud, which
    /// doesn't
    /// have nan point. If the value is false, return point cloud, which has
    /// a point for each pixel, whereas invalid depth results in NaN points.
    static std::shared_ptr<PointCloud> CreateFromRGBDImage(
            const RGBDImage &image,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4f &extrinsic = Eigen::Matrix4f::Identity(),
            bool project_valid_depth_only = true,
            float depth_cutoff = -1.0f,
            bool compute_normals = false);

    static std::shared_ptr<PointCloud> CreateFromLaserScanBuffer(
            const LaserScanBuffer &scan, float min_range, float max_range);

    static std::shared_ptr<PointCloud> CreateFromOccupancyGrid(
            const OccupancyGrid &occgrid);

    static std::shared_ptr<PointCloud> CreateFromDisparity(
            const Image &disp,
            const Image &color,
            const camera::PinholeCameraIntrinsic &left_intrinsic,
            const camera::PinholeCameraIntrinsic &right_intrinsic,
            float baseline);

public:
    utility::device_vector<Eigen::Vector3f> points_;
    utility::device_vector<Eigen::Vector3f> normals_;
    utility::device_vector<Eigen::Vector3f> colors_;
    utility::device_vector<Eigen::Matrix3f> covariances_;
};

}  // namespace geometry
}  // namespace cupoch