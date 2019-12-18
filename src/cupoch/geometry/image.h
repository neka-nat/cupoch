#pragma once
#include "cupoch/geometry/geometry2d.h"
#include <thrust/device_vector.h>

namespace cupoch {
namespace geometry {

/// \class Image
///
/// \brief The Image class stores image with customizable width, height, num of
/// channels and bytes per channel.
class Image : public Geometry2D {
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
        Equal,
        /// Weighted R, G, B channels: I = 0.299 * R + 0.587 * G + 0.114 * B.
        Weighted,
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

    Image();
    ~Image() override;

    Image &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector2f GetMinBound() const override;
    Eigen::Vector2f GetMaxBound() const override;

    thrust::host_vector<uint8_t> GetData() const;
    void SetData(const thrust::host_vector<uint8_t>& data);

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
    __host__ __device__
    Image &Prepare(int width,
                   int height,
                   int num_of_channels,
                   int bytes_per_channel) {
        width_ = width;
        height_ = height;
        num_of_channels_ = num_of_channels;
        bytes_per_channel_ = bytes_per_channel;
        AllocateDataBuffer();
        return *this;
    }

    /// \brief Returns data size per line (row, or the width) in bytes.
    __host__ __device__
    int BytesPerLine() const {
        return width_ * num_of_channels_ * bytes_per_channel_;
    }

    /// Function to flip image horizontally (from left to right).
    std::shared_ptr<Image> FlipHorizontal() const;
    /// Function to flip image vertically (upside down).
    std::shared_ptr<Image> FlipVertical() const;

protected:
    void AllocateDataBuffer();

public:
    /// Width of the image.
    int width_ = 0;
    /// Height of the image.
    int height_ = 0;
    /// Number of chanels in the image.
    int num_of_channels_ = 0;
    /// Number of bytes per channel.
    int bytes_per_channel_ = 0;
    /// Image storage buffer.
    thrust::device_vector<uint8_t> data_;
};

}
}