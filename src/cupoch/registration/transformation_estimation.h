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
    ColoredICP = 3,
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
    virtual Eigen::Matrix4f ComputeTransformation(
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
    TransformationEstimationPointToPlane() {}
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
            const geometry::PointCloud &source,
            const geometry::PointCloud &target,
            const CorrespondenceSet &corres) const override;

private:
    const TransformationEstimationType type_ =
            TransformationEstimationType::PointToPlane;
};

}
}