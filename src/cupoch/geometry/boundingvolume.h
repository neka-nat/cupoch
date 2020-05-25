#pragma once

#include <Eigen/Core>
#include <array>

#include "cupoch/geometry/geometry3d.h"
#include "cupoch/utility/helper.h"

namespace cupoch {
namespace geometry {

class AxisAlignedBoundingBox;

/// \class OrientedBoundingBox
///
/// \brief A bounding box oriented along an arbitrary frame of reference.
///
/// The oriented bounding box is defined by its center position, rotation
/// maxtrix and extent.
class OrientedBoundingBox : public Geometry3D {
public:
    /// \brief Default constructor.
    ///
    /// Creates an empty Oriented Bounding Box.
    __host__ __device__
    OrientedBoundingBox()
        : Geometry3D(Geometry::GeometryType::OrientedBoundingBox),
          center_(0, 0, 0),
          R_(Eigen::Matrix3f::Identity()),
          extent_(0, 0, 0),
          color_(0, 0, 0) {}
    /// \brief Parameterized constructor.
    ///
    /// \param center Specifies the center position of the bounding box.
    /// \param R The rotation matrix specifying the orientation of the
    /// bounding box with the original frame of reference.
    /// \param extent The extent of the bounding box.
    __host__ __device__
    OrientedBoundingBox(const Eigen::Vector3f &center,
                        const Eigen::Matrix3f &R,
                        const Eigen::Vector3f &extent)
        : Geometry3D(Geometry::GeometryType::OrientedBoundingBox),
          center_(center),
          R_(R),
          extent_(extent) {}
    __host__ __device__ ~OrientedBoundingBox() {}

public:
    OrientedBoundingBox &Clear() override;
    bool IsEmpty() const override;
    virtual Eigen::Vector3f GetMinBound() const override;
    virtual Eigen::Vector3f GetMaxBound() const override;
    virtual Eigen::Vector3f GetCenter() const override;
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    virtual OrientedBoundingBox GetOrientedBoundingBox() const;
    virtual OrientedBoundingBox &Transform(
            const Eigen::Matrix4f &transformation) override;
    virtual OrientedBoundingBox &Translate(const Eigen::Vector3f &translation,
                                           bool relative = true) override;
    virtual OrientedBoundingBox &Scale(const float scale,
                                       bool center = true) override;
    virtual OrientedBoundingBox &Rotate(const Eigen::Matrix3f &R,
                                        bool center = true) override;

    /// Returns the volume of the bounding box.
    float Volume() const;
    /// Returns the eight points that define the bounding box.
    std::array<Eigen::Vector3f, 8> GetBoxPoints() const;

    /// Return indices to points that are within the bounding box.
    utility::device_vector<size_t> GetPointIndicesWithinBoundingBox(
            const utility::device_vector<Eigen::Vector3f> &points) const;

    /// Returns an oriented bounding box from the AxisAlignedBoundingBox.
    ///
    /// \param aabox AxisAlignedBoundingBox object from which
    /// OrientedBoundingBox is created.
    static OrientedBoundingBox CreateFromAxisAlignedBoundingBox(
            const AxisAlignedBoundingBox &aabox);

public:
    /// The center point of the bounding box.
    Eigen::Vector3f center_;
    /// The rotation matrix of the bounding box to transform the original frame
    /// of reference to the frame of this box.
    Eigen::Matrix3f R_;
    /// The extent of the bounding box in its frame of reference.
    Eigen::Vector3f extent_;
    /// The color of the bounding box in RGB.
    Eigen::Vector3f color_;
};

/// \class AxisAlignedBoundingBox
///
/// \brief A bounding box that is aligned along the coordinate axes.
///
///  The AxisAlignedBoundingBox uses the cooridnate axes for bounding box
///  generation. This means that the bounding box is oriented along the
///  coordinate axes.
class AxisAlignedBoundingBox : public Geometry3D {
public:
    /// \brief Default constructor.
    ///
    /// Creates an empty Axis Aligned Bounding Box.
    __host__ __device__
    AxisAlignedBoundingBox()
        : Geometry3D(Geometry::GeometryType::AxisAlignedBoundingBox),
          min_bound_(0, 0, 0),
          max_bound_(0, 0, 0),
          color_(0, 0, 0) {}
    /// \brief Parameterized constructor.
    ///
    /// \param min_bound Lower bounds of the bounding box for all axes.
    /// \param max_bound Upper bounds of the bounding box for all axes.
    __host__ __device__
    AxisAlignedBoundingBox(const Eigen::Vector3f &min_bound,
                           const Eigen::Vector3f &max_bound)
        : Geometry3D(Geometry::GeometryType::AxisAlignedBoundingBox),
          min_bound_(min_bound),
          max_bound_(max_bound),
          color_(0, 0, 0) {}
    __host__ __device__ ~AxisAlignedBoundingBox() {}

public:
    AxisAlignedBoundingBox &Clear() override;
    bool IsEmpty() const override;
    virtual Eigen::Vector3f GetMinBound() const override;
    virtual Eigen::Vector3f GetMaxBound() const override;
    virtual Eigen::Vector3f GetCenter() const override;
    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    virtual OrientedBoundingBox GetOrientedBoundingBox() const;
    virtual AxisAlignedBoundingBox &Transform(
            const Eigen::Matrix4f &transformation) override;
    virtual AxisAlignedBoundingBox &Translate(
            const Eigen::Vector3f &translation, bool relative = true) override;
    virtual AxisAlignedBoundingBox &Scale(const float scale,
                                          bool center = true) override;
    virtual AxisAlignedBoundingBox &Rotate(const Eigen::Matrix3f &R,
                                           bool center = true) override;

    AxisAlignedBoundingBox &operator+=(const AxisAlignedBoundingBox &other);

    /// Get the extent/length of the bounding box in x, y, and z dimension.
    Eigen::Vector3f GetExtent() const { return (max_bound_ - min_bound_); }

    /// Returns the half extent of the bounding box.
    Eigen::Vector3f GetHalfExtent() const { return GetExtent() * 0.5; }

    /// Returns the maximum extent, i.e. the maximum of X, Y and Z axis'
    /// extents.
    float GetMaxExtent() const { return (max_bound_ - min_bound_).maxCoeff(); }

    __host__ __device__ float GetXPercentage(float x) const {
        return (x - min_bound_(0)) / (max_bound_(0) - min_bound_(0));
    }

    __host__ __device__ float GetYPercentage(float y) const {
        return (y - min_bound_(1)) / (max_bound_(1) - min_bound_(1));
    }

    __host__ __device__ float GetZPercentage(float z) const {
        return (z - min_bound_(2)) / (max_bound_(2) - min_bound_(2));
    }

    /// Returns the volume of the bounding box.
    float Volume() const;
    /// Returns the eight points that define the bounding box.
    std::array<Eigen::Vector3f, 8> GetBoxPoints() const;

    /// Return indices to points that are within the bounding box.
    ///
    /// \param points A list of points.
    utility::device_vector<size_t> GetPointIndicesWithinBoundingBox(
            const utility::device_vector<Eigen::Vector3f> &points) const;

    /// Returns the 3D dimensions of the bounding box in string format.
    std::string GetPrintInfo() const;

    /// Creates the bounding box that encloses the set of points.
    ///
    /// \param points A list of points.
    static AxisAlignedBoundingBox CreateFromPoints(
            const utility::device_vector<Eigen::Vector3f> &points);

public:
    /// The lower x, y, z bounds of the bounding box.
    Eigen::Vector3f min_bound_;
    /// The upper x, y, z bounds of the bounding box.
    Eigen::Vector3f max_bound_;
    /// The color of the bounding box in RGB.
    Eigen::Vector3f color_;
};

}  // namespace geometry
}  // namespace cupoch