#pragma once
#include "cupoch/geometry/geometry.h"
#include "cupoch/utility/eigen.h"
#include "cupoch/geometry/kdtree_search_param.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace cupoch {
namespace geometry {

class PointCloud : public Geometry {
public:
    PointCloud();
    PointCloud(const thrust::host_vector<Eigen::Vector3f>& points);
    PointCloud(const PointCloud& other);
    ~PointCloud();

    void SetPoints(const thrust::host_vector<Eigen::Vector3f>& points);
    thrust::host_vector<Eigen::Vector3f> GetPoints() const;

    void SetNormals(const thrust::host_vector<Eigen::Vector3f>& normals);
    thrust::host_vector<Eigen::Vector3f> GetNormals() const;

    void SetColors(const thrust::host_vector<Eigen::Vector3f>& colors);
    thrust::host_vector<Eigen::Vector3f> GetColors() const;

    Eigen::Vector3f GetMinBound() const;
    Eigen::Vector3f GetMaxBound() const;
    Eigen::Vector3f GetCenter() const;

    PointCloud &Clear() override;
    bool IsEmpty() const override;

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
    PointCloud &PaintUniformColor(const Eigen::Vector3f &color);

    PointCloud& Transform(const Eigen::Matrix4f& transformation);

    PointCloud &RemoveNoneFinitePoints(bool remove_nan = true,
                                       bool remove_infinite = true);

    std::shared_ptr<PointCloud> SelectDownSample(const thrust::device_vector<size_t> &indices, bool invert = false) const;

    std::shared_ptr<PointCloud> VoxelDownSample(float voxel_size) const;

    std::shared_ptr<PointCloud> UniformDownSample(size_t every_k_points) const;


    std::tuple<std::shared_ptr<PointCloud>, thrust::device_vector<size_t>>
    RemoveRadiusOutliers(size_t nb_points, float search_radius) const;

    std::tuple<std::shared_ptr<PointCloud>, thrust::host_vector<size_t>>
    RemoveRadiusOutliersHost(size_t nb_points, float search_radius) const;


    std::tuple<std::shared_ptr<PointCloud>, thrust::device_vector<size_t>>
    RemoveStatisticalOutliers(size_t nb_neighbors, float std_ratio) const;

    std::tuple<std::shared_ptr<PointCloud>, thrust::host_vector<size_t>>
    RemoveStatisticalOutliersHost(size_t nb_neighbors, float std_ratio) const;


    std::shared_ptr<PointCloud> Crop(const Eigen::Vector3f &min_bound,
                                     const Eigen::Vector3f &max_bound) const;

    bool EstimateNormals(const KDTreeSearchParam& search_param = KDTreeSearchParamKNN());
    bool OrientNormalsToAlignWithDirection(const Eigen::Vector3f &orientation_reference = Eigen::Vector3f(0.0, 0.0, 1.0));

public:
    thrust::device_vector<Eigen::Vector3f> points_;
    thrust::device_vector<Eigen::Vector3f> normals_;
    thrust::device_vector<Eigen::Vector3f> colors_;
};


}
}