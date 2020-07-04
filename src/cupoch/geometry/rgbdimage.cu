#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/rgbdimage.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::geometry;

RGBDImage &RGBDImage::Clear() {
    color_.Clear();
    depth_.Clear();
    return *this;
}

bool RGBDImage::IsEmpty() const {
    return !color_.HasData() || !depth_.HasData();
}

Eigen::Vector2f RGBDImage::GetMinBound() const {
    return Eigen::Vector2f(0.0, 0.0);
}

Eigen::Vector2f RGBDImage::GetMaxBound() const {
    return Eigen::Vector2f(color_.width_ + depth_.width_, color_.height_);
}

Eigen::Vector2f RGBDImage::GetCenter() const {
    return Eigen::Vector2f((color_.width_ + depth_.width_) / 2,
                           color_.height_ / 2);
}

AxisAlignedBoundingBox RGBDImage::GetAxisAlignedBoundingBox() const {
    utility::LogError("RGBDImage::GetAxisAlignedBoundingBox is not supported");
    return AxisAlignedBoundingBox();
}

RGBDImage &RGBDImage::Transform(const Eigen::Matrix3f &transformation) {
    utility::LogError("RGBDImage::Transform is not supported");
    return *this;
}

RGBDImage &RGBDImage::Translate(const Eigen::Vector2f &translation,
                                bool relative) {
    utility::LogError("RGBDImage::Translate is not supported");
    return *this;
}

RGBDImage &RGBDImage::Scale(const float scale, bool center) {
    utility::LogError("RGBDImage::Scale is not supported");
    return *this;
}

RGBDImage &RGBDImage::Rotate(const Eigen::Matrix2f &R, bool center) {
    utility::LogError("RGBDImage::Rotate is not supported");
    return *this;
}

RGBDImagePyramid RGBDImage::FilterPyramid(
        const RGBDImagePyramid &rgbd_image_pyramid, Image::FilterType type) {
    RGBDImagePyramid rgbd_image_pyramid_filtered;
    rgbd_image_pyramid_filtered.clear();
    int num_of_levels = (int)rgbd_image_pyramid.size();
    for (int level = 0; level < num_of_levels; level++) {
        auto color_level = rgbd_image_pyramid[level]->color_;
        auto depth_level = rgbd_image_pyramid[level]->depth_;
        auto color_level_filtered = color_level.Filter(type);
        auto depth_level_filtered = depth_level.Filter(type);
        auto rgbd_image_level_filtered = std::make_shared<RGBDImage>(
                RGBDImage(*color_level_filtered, *depth_level_filtered));
        rgbd_image_pyramid_filtered.push_back(rgbd_image_level_filtered);
    }
    return rgbd_image_pyramid_filtered;
}

RGBDImagePyramid RGBDImage::CreatePyramid(
        size_t num_of_levels,
        bool with_gaussian_filter_for_color /* = true */,
        bool with_gaussian_filter_for_depth /* = false */) const {
    ImagePyramid color_pyramid =
            color_.CreatePyramid(num_of_levels, with_gaussian_filter_for_color);
    ImagePyramid depth_pyramid =
            depth_.CreatePyramid(num_of_levels, with_gaussian_filter_for_depth);
    RGBDImagePyramid rgbd_image_pyramid;
    rgbd_image_pyramid.clear();
    for (size_t level = 0; level < num_of_levels; level++) {
        auto rgbd_image_level = std::make_shared<RGBDImage>(
                RGBDImage(*color_pyramid[level], *depth_pyramid[level]));
        rgbd_image_pyramid.push_back(rgbd_image_level);
    }
    return rgbd_image_pyramid;
}
