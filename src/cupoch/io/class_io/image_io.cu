#include "cupoch/io/class_io/image_io.h"
#include "cupoch/utility/helper.h"

using namespace cupoch;
using namespace cupoch::io;

void HostImage::FromDevice(const geometry::Image& image) {
    data_.resize(image.data_.size());
    utility::CopyFromDeviceMultiStream(image.data_, data_);
    cudaDeviceSynchronize();
    width_ = image.width_;
    height_ = image.height_;
    num_of_channels_ = image.num_of_channels_;
    bytes_per_channel_ = image.bytes_per_channel_;
}

void HostImage::ToDevice(geometry::Image& image) const {
    image.data_.resize(data_.size());
    utility::CopyToDeviceMultiStream(data_, image.data_);
    cudaDeviceSynchronize();
    image.width_ = width_;
    image.height_ = height_;
    image.num_of_channels_ = num_of_channels_;
    image.bytes_per_channel_ = bytes_per_channel_;
}

void HostImage::Clear() {
    data_.clear();
    width_ = 0;
    height_ = 0;
    num_of_channels_ = 0;
    bytes_per_channel_ = 0;
}

HostImage& HostImage::Prepare(int width,
    int height,
    int num_of_channels,
    int bytes_per_channel) {
    width_ = width;
    height_ = height;
    num_of_channels_ = num_of_channels;
    bytes_per_channel_ = bytes_per_channel;
    data_.resize(width_ * height_ * num_of_channels_ * bytes_per_channel_);
    return *this;
}