#include "cupoch/geometry/rgbdimage.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct convert_sun_format_functor {
    convert_sun_format_functor(uint8_t* depth) : depth_(depth) {};
    uint8_t* depth_;
    __device__
    void operator() (size_t idx) {
        uint16_t &d = *(uint16_t*)(depth_ + idx * sizeof(uint16_t));
        d = (d >> 3) | (d << 13);
    }
};

struct convert_nyu_format_functor {
    convert_nyu_format_functor(uint8_t* depth) : depth_(depth) {};
    uint8_t* depth_;
    __device__
    void operator() (size_t idx) {
        uint16_t *d = (uint16_t*)(depth_ + idx * sizeof(uint16_t));
        uint8_t *p = (uint8_t *)d;
        uint8_t x = *p;
        *p = *(p + 1);
        *(p + 1) = x;
        float xx = 351.3 / (1092.5 - *d);
        if (xx <= 0.0) {
            *d = 0;
        } else {
            *d = (uint16_t)(floor(xx * 1000 + 0.5));
        }
    }
};

}

std::shared_ptr<RGBDImage> RGBDImage::CreateFromColorAndDepth(
        const Image &color,
        const Image &depth,
        double depth_scale /* = 1000.0*/,
        double depth_trunc /* = 3.0*/,
        bool convert_rgb_to_intensity /* = true*/) {
    std::shared_ptr<RGBDImage> rgbd_image = std::make_shared<RGBDImage>();
    if (color.height_ != depth.height_ || color.width_ != depth.width_) {
        utility::LogError(
                "[CreateFromColorAndDepth] Unsupported image "
                "format.");
    }
    rgbd_image->depth_ =
            *depth.ConvertDepthToFloatImage(depth_scale, depth_trunc);
    rgbd_image->color_ =
            convert_rgb_to_intensity ? *color.CreateFloatImage() : color;
    return rgbd_image;
}

/// Reference: http://redwood-data.org/indoor/
/// File format: http://redwood-data.org/indoor/dataset.html
std::shared_ptr<RGBDImage> RGBDImage::CreateFromRedwoodFormat(
        const Image &color,
        const Image &depth,
        bool convert_rgb_to_intensity /* = true*/) {
    return CreateFromColorAndDepth(color, depth, 1000.0, 4.0,
                                   convert_rgb_to_intensity);
}

/// Reference: http://vision.in.tum.de/data/datasets/rgbd-dataset
/// File format: http://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
std::shared_ptr<RGBDImage> RGBDImage::CreateFromTUMFormat(
        const Image &color,
        const Image &depth,
        bool convert_rgb_to_intensity /* = true*/) {
    return CreateFromColorAndDepth(color, depth, 5000.0, 4.0,
                                   convert_rgb_to_intensity);
}

/// Reference: http://sun3d.cs.princeton.edu/
/// File format: https://github.com/PrincetonVision/SUN3DCppReader
std::shared_ptr<RGBDImage> RGBDImage::CreateFromSUNFormat(
        const Image &color,
        const Image &depth,
        bool convert_rgb_to_intensity /* = true*/) {
    if (color.height_ != depth.height_ || color.width_ != depth.width_) {
        utility::LogError(
                "[CreateRGBDImageFromSUNFormat] Unsupported image format.");
    }
    std::shared_ptr<Image> depth_t = std::make_shared<Image>();
    *depth_t = depth;
    convert_sun_format_functor func(thrust::raw_pointer_cast(depth_t->data_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(depth_t->width_ * depth_t->height_), func);
    // SUN depth map has long range depth. We set depth_trunc as 7.0
    return CreateFromColorAndDepth(color, *depth_t, 1000.0, 7.0,
                                   convert_rgb_to_intensity);
}

/// Reference: http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
std::shared_ptr<RGBDImage> RGBDImage::CreateFromNYUFormat(
        const Image &color,
        const Image &depth,
        bool convert_rgb_to_intensity /* = true*/) {
    if (color.height_ != depth.height_ || color.width_ != depth.width_) {
        utility::LogError(
                "[CreateRGBDImageFromNYUFormat] Unsupported image format.");
    }
    std::shared_ptr<Image> depth_t = std::make_shared<Image>();
    *depth_t = depth;
    convert_nyu_format_functor func(thrust::raw_pointer_cast(depth_t->data_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(depth_t->width_ * depth_t->height_), func);
    // NYU depth map has long range depth. We set depth_trunc as 7.0
    return CreateFromColorAndDepth(color, *depth_t, 1000.0, 7.0,
                                   convert_rgb_to_intensity);
}