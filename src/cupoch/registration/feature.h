#pragma once

#include <Eigen/Core>
#include <memory>
#include "cupoch/utility/device_vector.h"

#include "cupoch/geometry/kdtree_search_param.h"

namespace cupoch {

namespace geometry {
class PointCloud;
}

namespace registration {

template<int Dim>
class Feature {
public:
    void Resize(int n) { data_.resize(n); };
    size_t Dimension() const { return Dim; }
    size_t Num() const { return data_.size(); };
public:
    typedef Eigen::Matrix<float, Dim, 1> FeatureType;
    utility::device_vector<FeatureType> data_;
};

/// Function to compute FPFH feature for a point cloud
std::shared_ptr<Feature<33>> ComputeFPFHFeature(
        const geometry::PointCloud &input,
        const geometry::KDTreeSearchParam &search_param =
                geometry::KDTreeSearchParamKNN());

}  // namespace registration
}  // namespace cupoch