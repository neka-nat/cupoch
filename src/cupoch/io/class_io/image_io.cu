#include "cupoch/io/class_io/image_io.h"
#include "cupoch/utility/helper.h"

using namespace cupoch;
using namespace cupoch::io;

void HostImage::FromDevice(const geometry::Image& image) {
    data_.resize(image.data_.size());
    utility::CopyFromDeviceMultiStream(image.data_, data_);
    cudaDeviceSynchronize();
}

void HostImage::ToDevice(geometry::Image& image) const {
    image.data_.resize(data_.size());
    utility::CopyToDeviceMultiStream(data_, image.data_);
    cudaDeviceSynchronize();
}

void HostImage::Clear() {
    data_.clear();
}