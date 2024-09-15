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

#include <Eigen/Core>
#include <memory>

#include "cupoch/registration/transformation_estimation.h"
#include "cupoch/knn/kdtree_search_param.h"
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
    bool IsEmpty() const;
    std::vector<Eigen::Matrix<float, Dim, 1>> GetData() const;
    void SetData(const std::vector<Eigen::Matrix<float, Dim, 1>>& data);

public:
    typedef Eigen::Matrix<float, Dim, 1> FeatureType;
    utility::device_vector<FeatureType> data_;
};

/// Function to compute FPFH feature for a point cloud
std::shared_ptr<Feature<33>> ComputeFPFHFeature(
        const geometry::PointCloud& input,
        const knn::KDTreeSearchParam& search_param =
                knn::KDTreeSearchParamKNN());

/// Function to compute SHOT feature for a point cloud
std::shared_ptr<Feature<352>> ComputeSHOTFeature(
        const geometry::PointCloud& input,
        float radius,
        const knn::KDTreeSearchParam& search_param =
                knn::KDTreeSearchParamKNN());

CorrespondenceSet CorrespondencesFromFeatures(const Feature<33> &source_features,
                                              const Feature<33> &target_features,
                                              bool mutual_filter = false,
                                              float mutual_consistency_ratio = 0.1);

}  // namespace registration
}  // namespace cupoch