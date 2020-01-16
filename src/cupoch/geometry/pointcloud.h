#pragma once
#include "cupoch/geometry/geometry3d.h"
#include "cupoch/utility/eigen.h"
#include "cupoch/geometry/kdtree_search_param.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace cupoch {

namespace camera {
class PinholeCameraIntrinsic;
}

namespace geometry {

class Image;

class PointCloud : public Geometry3D {
public:
    PointCloud();
    PointCloud(const thrust::host_vector<Eigen::Vector3f>& points);
    PointCloud(const PointCloud& other);
    ~PointCloud();
    PointCloud& operator=(const PointCloud& other);

    void SetPoints(const thrust::host_vector<Eigen::Vector3f>& points);
    thrust::host_vector<Eigen::Vector3f> GetPoints() const;

    void SetNormals(const thrust::host_vector<Eigen::Vector3f>& normals);
    thrust::host_vector<Eigen::Vector3f> GetNormals() const;

    void SetColors(const thrust::host_vector<Eigen::Vector3f>& colors);
    thrust::host_vector<Eigen::Vector3f> GetColors() const;

    PointCloud &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3f GetMinBound() const override;
    Eigen::Vector3f GetMaxBound() const override;
    Eigen::Vector3f GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    PointCloud& Transform(const Eigen::Matrix4f& transformation) override;
    PointCloud& Translate(const Eigen::Vector3f &translation,
                          bool relative = true) override;
    PointCloud& Scale(const float scale, bool center = true) override;
    PointCloud& Rotate(const Eigen::Matrix3f &R, bool center = true) override;

    __host__ __device__
    bool HasPoints() const { return !points_.empty(); }

    __host__ __device__
    bool HasNormals() const {
        return !points_.empty() && normals_.size() == points_.size();
    }

    __host__ __device__
    bool HasColors() const {
        return !points_.empty() && colors_.size() == points_.size();
    }

    PointCloud &NormalizeNormals();

    /// Assigns each point in the PointCloud the same color \param color.
    PointCloud &PaintUniformColor(const Eigen::Vector3f &color);

    /// Remove all points fromt he point cloud that have a nan entry, or
    /// infinite entries.
    /// Also removes the corresponding normals and color entries.
    PointCloud &RemoveNoneFinitePoints(bool remove_nan = true,
                                       bool remove_infinite = true);

    /// Function to select points from \param input pointcloud into
    /// \return output pointcloud
    /// Points with indices in \param indices are selected.
    std::shared_ptr<PointCloud> SelectDownSample(const thrust::device_vector<size_t> &indices, bool invert = false) const;

    /// Function to downsample \param input pointcloud into output pointcloud
    /// with a voxel \param voxel_size defines the resolution of the voxel grid,
    /// smaller value leads to denser output point cloud. Normals and colors are
    /// averaged if they exist.
    std::shared_ptr<PointCloud> VoxelDownSample(float voxel_size) const;

    /// Function to downsample \param input pointcloud into output pointcloud
    /// uniformly \param every_k_points indicates the sample rate.
    std::shared_ptr<PointCloud> UniformDownSample(size_t every_k_points) const;


    std::tuple<std::shared_ptr<PointCloud>, thrust::device_vector<size_t>>
    RemoveRadiusOutliers(size_t nb_points, float search_radius) const;

    std::tuple<std::shared_ptr<PointCloud>, thrust::host_vector<size_t>>
    RemoveRadiusOutliersHost(size_t nb_points, float search_radius) const;


    std::tuple<std::shared_ptr<PointCloud>, thrust::device_vector<size_t>>
    RemoveStatisticalOutliers(size_t nb_neighbors, float std_ratio) const;

    std::tuple<std::shared_ptr<PointCloud>, thrust::host_vector<size_t>>
    RemoveStatisticalOutliersHost(size_t nb_neighbors, float std_ratio) const;


    /// Function to crop pointcloud into output pointcloud
    /// All points with coordinates outside the bounding box \param bbox are
    /// clipped.
    std::shared_ptr<PointCloud> Crop(const Eigen::Vector3f &min_bound,
                                     const Eigen::Vector3f &max_bound) const;

    /// Function to compute the normals of a point cloud
    /// \param cloud is the input point cloud. It also stores the output
    /// normals. Normals are oriented with respect to the input point cloud if
    /// normals exist in the input. \param search_param The KDTree search
    /// parameters
    bool EstimateNormals(const KDTreeSearchParam& search_param = KDTreeSearchParamKNN());

    /// Function to orient the normals of a point cloud
    /// \param cloud is the input point cloud. It must have normals.
    /// Normals are oriented with respect to \param orientation_reference
    bool OrientNormalsToAlignWithDirection(const Eigen::Vector3f &orientation_reference = Eigen::Vector3f(0.0, 0.0, 1.0));

    /// Cluster PointCloud using the DBSCAN algorithm
    /// Ester et al., "A Density-Based Algorithm for Discovering Clusters
    /// in Large Spatial Databases with Noise", 1996
    /// Returns a vector of point labels, -1 indicates noise according to
    /// the algorithm.
    thrust::device_vector<int> ClusterDBSCAN(float eps,
                                             size_t min_points,
                                             bool print_progress = false) const;
    thrust::host_vector<int> ClusterDBSCANHost(float eps,
                                               size_t min_points,
                                               bool print_progress = false) const;

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

public:
    thrust::device_vector<Eigen::Vector3f> points_;
    thrust::device_vector<Eigen::Vector3f> normals_;
    thrust::device_vector<Eigen::Vector3f> colors_;
};


}
}