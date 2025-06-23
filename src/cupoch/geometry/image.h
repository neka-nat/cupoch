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
#include <vector>

#include "cupoch/geometry/geometry_base.h"
#include "cupoch/utility/device_vector.h"

namespace cupoch {

namespace camera {
class PinholeCameraIntrinsic;
}

namespace geometry {

class Image;

/// Typedef and functions for ImagePyramid.
typedef std::vector<std::shared_ptr<Image>> ImagePyramid;

/// \class Image
///
/// \brief The Image class stores image with customizable width, height, num of
/// channels and bytes per channel.
class Image : public GeometryBaseNoTrans2D {
public:
    /// \enum ColorToIntensityConversionType
    ///
    /// \brief Specifies whether R, G, B channels have the same weight when
    /// converting to intensity. Only used for Image with 3 channels.
    ///
    /// When `Weighted` is used R, G, B channels are weighted according to the
    /// Digital ITU BT.601 standard: I = 0.299 * R + 0.587 * G + 0.114 * B.
    enum class ColorToIntensityConversionType {
        /// R, G, B channels have equal weights.
        Equal = 0,
        /// Weighted R, G, B channels: I = 0.299 * R + 0.587 * G + 0.114 * B.
        Weighted = 1,
    };

    /// \enum FilterType
    ///
    /// \brief Specifies the Image filter type.
    enum class FilterType {
        /// Gaussian filter of size 3 x 3.
        Gaussian3,
        /// Gaussian filter of size 5 x 5.
        Gaussian5,
        /// Gaussian filter of size 7 x 7.
        Gaussian7,
        /// Sobel filter along X-axis.
        Sobel3Dx,
        /// Sobel filter along Y or.
        Sobel3Dy
    };

public:
    Image();
    ~Image();
    Image(const Image &other);
    Image &operator=(const Image &other);

    Image &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector2f GetMinBound() const override;
    Eigen::Vector2f GetMaxBound() const override;
    Eigen::Vector2f GetCenter() const override;

    std::vector<uint8_t> GetData() const;
    void SetData(const thrust::host_vector<uint8_t> &data);
    void SetData(const std::vector<uint8_t> &data);

    /// \brief Test if coordinate `(u, v)` is located in the inner_marge of the
    /// image.
    ///
    /// \param u Coordinate along the width dimension.
    /// \param v Coordinate along the height dimension.
    /// \param inner_margin The inner margin from the image boundary.
    /// \return Returns `true` if coordinate `(u, v)` is located in the
    /// inner_marge of the image.
    bool TestImageBoundary(float u, float v, float inner_margin = 0.0) const;

    /// Returns `true` if the Image has valid data.
    virtual bool HasData() const {
        return width_ > 0 && height_ > 0 &&
               data_.size() == size_t(height_ * BytesPerLine());
    }

    /// \brief Prepare Image properties and allocate Image buffer.
     Image &Prepare(unsigned int width,
                    unsigned int height,
                    unsigned int num_of_channels,
                    unsigned int bytes_per_channel) {
        width_ = width;
        height_ = height;
        num_of_channels_ = num_of_channels;
        bytes_per_channel_ = bytes_per_channel;
        AllocateDataBuffer();
        return *this;
    }

    /// \brief Returns data size per line (row, or the width) in bytes.
    __host__ __device__ int BytesPerLine() const {
        return width_ * num_of_channels_ * bytes_per_channel_;
    }

    /// Function to access the bilinear interpolated float value of a
    /// (single-channel) float image.
    /// Returns a tuple, where the first bool indicates if the u,v coordinates
    /// are within the image dimensions, and the second float value is the
    /// interpolated pixel value.
    std::pair<bool, float> FloatValueAt(float u, float v) const;

    /// Factory function to create a float image composed of multipliers that
    /// convert depth values into camera distances (ImageFactory.cpp)
    /// The multiplier function M(u,v) is defined as:
    /// M(u, v) = sqrt(1 + ((u - cx) / fx) ^ 2 + ((v - cy) / fy) ^ 2)
    /// This function is used as a convenient function for performance
    /// optimization in volumetric integration (see
    /// Core/Integration/TSDFVolume.h).
    static std::shared_ptr<Image>
    CreateDepthToCameraDistanceMultiplierFloatImage(
            const camera::PinholeCameraIntrinsic &intrinsic);

    /// Return a gray scaled float type image.
    std::shared_ptr<Image> CreateGrayImage(
            Image::ColorToIntensityConversionType type =
                    Image::ColorToIntensityConversionType::Weighted) const;

    /// Return a gray scaled float type image.
    std::shared_ptr<Image> CreateFloatImage(
            Image::ColorToIntensityConversionType type =
                    Image::ColorToIntensityConversionType::Weighted) const;

    std::shared_ptr<Image> ConvertDepthToFloatImage(
            float depth_scale = 1000.0, float depth_trunc = 3.0) const;

    std::shared_ptr<Image> Transpose() const;

    /// Function to flip image horizontally (from left to right).
    std::shared_ptr<Image> FlipHorizontal() const;
    /// Function to flip image vertically (upside down).
    std::shared_ptr<Image> FlipVertical() const;

    /// Function to filter image with pre-defined filtering type.
    std::shared_ptr<Image> Filter(Image::FilterType type) const;

    /// Function to filter image with arbitrary dx, dy separable filters.
    std::shared_ptr<Image> Filter(
            const utility::device_vector<float> &dx,
            const utility::device_vector<float> &dy) const;

    std::shared_ptr<Image> FilterHorizontal(
            const utility::device_vector<float> &kernel) const;

    std::shared_ptr<Image> FilterHorizontal(
            const std::vector<float> &kernel) const;

    std::shared_ptr<Image> BilateralFilter(int diameter,
                                           float sigma_color,
                                           float sigma_space) const;

    /// Function to 2x image downsample using simple 2x2 averaging.
    std::shared_ptr<Image> Downsample() const;

    /// Function to linearly transform pixel intensities
    /// image_new = scale * image + offset.
    Image &LinearTransform(float scale = 1.0, float offset = 0.0);

    /// Function to clipping pixel intensities.
    ///
    /// \param min is lower bound.
    /// \param max is upper bound.
    Image &ClipIntensity(float min = 0.0, float max = 1.0);

    /// Function to change data types of image
    /// crafted for specific usage such as
    /// single channel float image -> 8-bit RGB or 16-bit depth image.
    template <typename T>
    std::shared_ptr<Image> CreateImageFromFloatImage() const;

    /// Function to filter image pyramid.
    static ImagePyramid FilterPyramid(const ImagePyramid &input,
                                      Image::FilterType type);
    static ImagePyramid BilateralFilterPyramid(const ImagePyramid &input,
                                               int diameter,
                                               float sigma_color,
                                               float sigma_space);

    /// Function to create image pyramid.
    ImagePyramid CreatePyramid(size_t num_of_levels,
                               bool with_gaussian_filter = true) const;

protected:
    void AllocateDataBuffer();

public:
    /// Width of the image.
    unsigned int width_ = 0;
    /// Height of the image.
    unsigned int height_ = 0;
    /// Number of chanels in the image.
    unsigned int num_of_channels_ = 0;
    /// Number of bytes per channel.
    unsigned int bytes_per_channel_ = 0;
    /// Image storage buffer.
    utility::device_vector<uint8_t> data_;
};

template <typename T>
__host__ __device__ T *PointerAt(const uint8_t *data, int width, int u, int v) {
    return (T *)(data + (v * width + u) * sizeof(T));
}

template <typename T>
__host__ __device__ T *PointerAt(const uint8_t *data,
                                 int width,
                                 int num_of_channels,
                                 int u,
                                 int v,
                                 int ch) {
    return (T *)(data + ((v * width + u) * num_of_channels + ch) * sizeof(T));
}

__host__ __device__ inline thrust::pair<bool, float> FloatValueAt(
        const uint8_t *data,
        float u,
        float v,
        int width,
        int height,
        int num_of_channels,
        int bytes_per_channel) {
    if ((num_of_channels != 1) || (bytes_per_channel != 4) ||
        (u < 0.0 || u > (float)(width - 1) || v < 0.0 ||
         v > (float)(height - 1))) {
        return thrust::make_pair(false, 0.0);
    }
    int ui = std::max(std::min((int)u, width - 2), 0);
    int vi = std::max(std::min((int)v, height - 2), 0);
    float pu = u - ui;
    float pv = v - vi;
    float value[4] = {*PointerAt<float>(data, width, ui, vi),
                      *PointerAt<float>(data, width, ui, vi + 1),
                      *PointerAt<float>(data, width, ui + 1, vi),
                      *PointerAt<float>(data, width, ui + 1, vi + 1)};
    return thrust::make_pair(
            true, (value[0] * (1 - pv) + value[1] * pv) * (1 - pu) +
                          (value[2] * (1 - pv) + value[3] * pv) * pu);
}

}  // namespace geometry
}  // namespace cupoch