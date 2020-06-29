#pragma once

#include <vector>

#include "cupoch/geometry/geometry_base.h"
#include "cupoch/geometry/image.h"

namespace cupoch {
namespace geometry {

class AxisAlignedBoundingBox;

class RGBDImage;

/// Typedef and functions for RGBDImagePyramid
typedef std::vector<std::shared_ptr<RGBDImage>> RGBDImagePyramid;

/// RGBDImage is for a pair of registered color and depth images,
/// viewed from the same view, of the same resolution.
/// If you have other format, convert it first.
class RGBDImage : public GeometryBase<2> {
public:
    RGBDImage() : GeometryBase<2>(Geometry::GeometryType::RGBDImage) {}
    RGBDImage(const Image &color, const Image &depth)
        : GeometryBase<2>(Geometry::GeometryType::RGBDImage),
          color_(color),
          depth_(depth) {}

    ~RGBDImage() {
        color_.Clear();
        depth_.Clear();
    };

    RGBDImage &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector2f GetMinBound() const override;
    Eigen::Vector2f GetMaxBound() const override;
    Eigen::Vector2f GetCenter() const override;
    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
    RGBDImage &Transform(const Eigen::Matrix3f &transformation) override;
    RGBDImage &Translate(const Eigen::Vector2f &translation, bool relative = true) override;
    RGBDImage &Scale(const float scale, bool center = true) override;
    RGBDImage &Rotate(const Eigen::Matrix2f &R, bool center = true) override;

    /// Factory function to create an RGBD Image from color and depth Images
    static std::shared_ptr<RGBDImage> CreateFromColorAndDepth(
            const Image &color,
            const Image &depth,
            float depth_scale = 1000.0,
            float depth_trunc = 3.0,
            bool convert_rgb_to_intensity = true);

    /// Factory function to create an RGBD Image from Redwood dataset
    static std::shared_ptr<RGBDImage> CreateFromRedwoodFormat(
            const Image &color,
            const Image &depth,
            bool convert_rgb_to_intensity = true);

    /// Factory function to create an RGBD Image from TUM dataset
    static std::shared_ptr<RGBDImage> CreateFromTUMFormat(
            const Image &color,
            const Image &depth,
            bool convert_rgb_to_intensity = true);

    /// Factory function to create an RGBD Image from SUN3D dataset
    static std::shared_ptr<RGBDImage> CreateFromSUNFormat(
            const Image &color,
            const Image &depth,
            bool convert_rgb_to_intensity = true);

    /// Factory function to create an RGBD Image from NYU dataset
    static std::shared_ptr<RGBDImage> CreateFromNYUFormat(
            const Image &color,
            const Image &depth,
            bool convert_rgb_to_intensity = true);

    static RGBDImagePyramid FilterPyramid(
            const RGBDImagePyramid &rgbd_image_pyramid, Image::FilterType type);

    RGBDImagePyramid CreatePyramid(
            size_t num_of_levels,
            bool with_gaussian_filter_for_color = true,
            bool with_gaussian_filter_for_depth = false) const;

public:
    Image color_;
    Image depth_;
};

}  // namespace geometry
}  // namespace cupoch