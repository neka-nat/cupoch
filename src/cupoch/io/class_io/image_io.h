#pragma once

#include <string>
#include <cupoch/utility/device_vector.h>

namespace cupoch {

namespace geometry {
class Image;
}

namespace io {

class HostImage {
public:
    HostImage() = default;
    ~HostImage() = default;
    void FromDevice(const geometry::Image& image);
    void ToDevice(geometry::Image& image) const;
    void Clear();
    HostImage& Prepare(int width,
                       int height,
                       int num_of_channels,
                       int bytes_per_channel);
    int BytesPerLine() const {
        return width_ * num_of_channels_ * bytes_per_channel_;
    }
    int width_ = 0;
    int height_ = 0;
    int num_of_channels_ = 0;
    int bytes_per_channel_ = 0;
    utility::pinned_host_vector<uint8_t> data_;
};

/// Factory function to create an image from a file (ImageFactory.cpp)
/// Return an empty image if fail to read the file.
std::shared_ptr<geometry::Image> CreateImageFromFile(
        const std::string &filename);

/// The general entrance for reading an Image from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ReadImage(const std::string &filename, geometry::Image &image);

/// The general entrance for writing an Image to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports quality, the parameter will be used.
/// Otherwise it will be ignored.
/// \return return true if the write function is successful, false otherwise.
bool WriteImage(const std::string &filename,
                const geometry::Image &image,
                int quality = 90);
bool WriteImage(const std::string &filename,
                const HostImage &image,
                int quality = 90);

bool ReadImageFromPNG(const std::string &filename, geometry::Image &image);

bool WriteImageToPNG(const std::string &filename,
                     const geometry::Image &image,
                     int quality);
bool WriteHostImageToPNG(const std::string &filename,
                         const HostImage &image,
                         int quality);

bool ReadImageFromJPG(const std::string &filename, geometry::Image &image);

bool WriteImageToJPG(const std::string &filename,
                     const geometry::Image &image,
                     int quality = 90);

}
}