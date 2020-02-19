#pragma once

#include "cupoch/geometry/geometry2d.h"
#include "cupoch/geometry/image.h"
#include <vector>

namespace cupoch {
namespace geometry {

class RGBDImage;

/// Typedef and functions for RGBDImagePyramid
typedef std::vector<std::shared_ptr<RGBDImage>> RGBDImagePyramid;

/// RGBDImage is for a pair of registered color and depth images,
/// viewed from the same view, of the same resolution.
/// If you have other format, convert it first.
class RGBDImage : public Geometry2D {
public:
    RGBDImage() : Geometry2D(Geometry::GeometryType::RGBDImage) {}
    RGBDImage(const Image &color, const Image &depth)
        : Geometry2D(Geometry::GeometryType::RGBDImage),
          color_(color),
          depth_(depth) {}

    ~RGBDImage() override {
        color_.Clear();
        depth_.Clear();
    };

    RGBDImage &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector2f GetMinBound() const override;
    Eigen::Vector2f GetMaxBound() const override;

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
}  // namespace open3d