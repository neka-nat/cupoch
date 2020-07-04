#pragma once

#include <Eigen/Core>
#include <memory>

#include "cupoch/geometry/kdtree_search_param.h"
#include "cupoch/utility/device_vector.h"

namespace cupoch {

namespace geometry {
class PointCloud;
}

namespace registration {

template <int Dim>
class Feature {
public:
    Feature();
    Feature(const Feature<Dim>& other);
    ~Feature();
    void Resize(int n);
    size_t Dimension() const;
    size_t Num() const;
    thrust::host_vector<Eigen::Matrix<float, Dim, 1>> GetData() const;
    void SetData(const thrust::host_vector<Eigen::Matrix<float, Dim, 1>>& data);

public:
    typedef Eigen::Matrix<float, Dim, 1> FeatureType;
    utility::device_vector<FeatureType> data_;
};

/// Function to compute FPFH feature for a point cloud
std::shared_ptr<Feature<33>> ComputeFPFHFeature(
        const geometry::PointCloud& input,
        const geometry::KDTreeSearchParam& search_param =
                geometry::KDTreeSearchParamKNN());

}  // namespace registration
}  // namespace cupoch