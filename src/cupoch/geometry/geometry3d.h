#pragma once
#include "cupoch/geometry/geometry.h"
#include "cupoch/utility/eigen.h"
#include <thrust/device_vector.h>

namespace cupoch {
namespace geometry {

class Geometry3D : public Geometry {
public:
    ~Geometry3D() override {}

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type type of object based on GeometryType.
    Geometry3D(GeometryType type) : Geometry(type, 3) {}

public:
    Eigen::Vector3f ComputeMinBound(const thrust::device_vector<Eigen::Vector3f>& points) const;

    Eigen::Vector3f ComputeMaxBound(const thrust::device_vector<Eigen::Vector3f>& points) const;

    Eigen::Vector3f ComuteCenter(const thrust::device_vector<Eigen::Vector3f>& points) const;

    void ResizeAndPaintUniformColor(thrust::device_vector<Eigen::Vector3f>& colors,
                                    const size_t size,
                                    const Eigen::Vector3f& color);

    void TransformPoints(const Eigen::Matrix4f& transformation,
                         thrust::device_vector<Eigen::Vector3f>& points);
    
    void TransformPoints(cudaStream_t stream, const Eigen::Matrix4f& transformation,
                         thrust::device_vector<Eigen::Vector3f>& points);
    
    void TransformNormals(const Eigen::Matrix4f& transformation,
                          thrust::device_vector<Eigen::Vector3f>& normals);
    
    void TransformNormals(cudaStream_t stream, const Eigen::Matrix4f& transformation,
                          thrust::device_vector<Eigen::Vector3f>& normals);
};

}
}