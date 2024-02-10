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

#include "cupoch/utility/device_vector.h"

namespace cupoch {

namespace geometry {
class PointCloud;
}

namespace registration {

typedef utility::device_vector<Eigen::Vector2i> CorrespondenceSet;

enum class TransformationEstimationType {
    Unspecified = 0,
    PointToPoint = 1,
    PointToPlane = 2,
    SymmetricMethod = 3,
    ColoredICP = 4,
    GeneralizedICP = 5,
};

/// Base class that estimates a transformation between two point clouds
/// The virtual function ComputeTransformation() must be implemented in
/// subclasses.
class TransformationEstimation {
public:
    TransformationEstimation() {}
    virtual ~TransformationEstimation() {}

public:
    virtual TransformationEstimationType GetTransformationEstimationType()
            const = 0;
    virtual float ComputeRMSE(const geometry::PointCloud &source,
                              const geometry::PointCloud &target,
                              const CorrespondenceSet &corres) const = 0;
    Eigen::Matrix4f ComputeTransformation(
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const;
    virtual Eigen::Matrix4f ComputeTransformation(
            cudaStream_t stream1, cudaStream_t stream2,
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const = 0;
};

/// Estimate a transformation for point to point distance
class TransformationEstimationPointToPoint : public TransformationEstimation {
public:
    TransformationEstimationPointToPoint() {}
    ~TransformationEstimationPointToPoint() override {}

public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    float ComputeRMSE(const geometry::PointCloud &source,
                      const geometry::PointCloud &target,
                      const CorrespondenceSet &corres) const override;
    Eigen::Matrix4f ComputeTransformation(
            cudaStream_t stream1, cudaStream_t stream2,
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPoint;
};

/// Estimate a transformation for point to plane distance
class TransformationEstimationPointToPlane : public TransformationEstimation {
public:
    TransformationEstimationPointToPlane(float det_thresh = 1.0e-6)
        : det_thresh_(det_thresh) {}
    ~TransformationEstimationPointToPlane() override {}

public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    float ComputeRMSE(const geometry::PointCloud &source,
                      const geometry::PointCloud &target,
                      const CorrespondenceSet &corres) const override;
    Eigen::Matrix4f ComputeTransformation(
            cudaStream_t stream1, cudaStream_t stream2,
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

    float det_thresh_;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPlane;
};

/// Estimate a transformation for plane to plane distance
class TransformationEstimationSymmetricMethod : public TransformationEstimation {
public:
    TransformationEstimationSymmetricMethod(float det_thresh = 1.0e-6)
        : det_thresh_(det_thresh) {}
    ~TransformationEstimationSymmetricMethod() override {}

public:
    TransformationEstimationType GetTransformationEstimationType()
            const override {
        return type_;
    };
    float ComputeRMSE(const geometry::PointCloud &source,
                      const geometry::PointCloud &target,
                      const CorrespondenceSet &corres) const override;
    Eigen::Matrix4f ComputeTransformation(
            cudaStream_t stream1, cudaStream_t stream2,
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

    float det_thresh_;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::SymmetricMethod;
};

}  // namespace registration
}  // namespace cupoch