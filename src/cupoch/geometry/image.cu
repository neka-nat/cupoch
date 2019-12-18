#include "cupoch/geometry/image.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct vertical_flip_functor {
    vertical_flip_functor(const uint8_t* src, int width, int height,
                          int bytes_per_pixel, uint8_t* dst)
        : src_(src), width_(width), height_(height), bytes_per_pixel_(bytes_per_pixel), dst_(dst) {};
    const uint8_t* src_;
    const int width_;
    const int height_;
    const int bytes_per_pixel_;
    uint8_t* dst_;
    __device__
    void operator() (size_t idx) {
        const int y = idx / width_;
        const int x = idx % width_;
        memcpy(&dst_[(height_ - y - 1) * width_ * bytes_per_pixel_ + x], &src_[idx], bytes_per_pixel_ * sizeof(uint8_t));
    }
};

struct horizontal_flip_functor {
    horizontal_flip_functor(const uint8_t* src, int width, int height,
                            int bytes_per_pixel, uint8_t* dst)
        : src_(src), width_(width), height_(height), bytes_per_pixel_(bytes_per_pixel), dst_(dst) {};
    const uint8_t* src_;
    const int width_;
    const int height_;
    const int bytes_per_pixel_;
    uint8_t* dst_;
    __device__
    void operator() (size_t idx) {
        const int y = idx / width_;
        const int x = idx % width_;
        memcpy(&dst_[y * width_ * bytes_per_pixel_ + (width_ - x - 1)], &src_[idx], bytes_per_pixel_ * sizeof(uint8_t));
    }
};

}


Image::Image() : Geometry2D(Geometry::GeometryType::Image) {}
Image::~Image() {}

Image &Image::Clear() {
    width_ = 0;
    height_ = 0;
    num_of_channels_ = 0;
    bytes_per_channel_ = 0;
    data_.clear();
    return *this;
}

bool Image::IsEmpty() const {
    return !HasData();
}

Eigen::Vector2f Image::GetMinBound() const {
    return Eigen::Vector2f(0.0, 0.0);
}

Eigen::Vector2f Image::GetMaxBound() const {
    return Eigen::Vector2f(width_, height_);
}

thrust::host_vector<uint8_t> Image::GetData() const {
    thrust::host_vector<uint8_t> data = data_;
    return data;
}

void Image::SetData(const thrust::host_vector<uint8_t>& data) {
    data_ = data;
}

bool Image::TestImageBoundary(float u,
                              float v,
                              float inner_margin /* = 0.0 */) const {
    return (u >= inner_margin && u < width_ - inner_margin &&
            v >= inner_margin && v < height_ - inner_margin);
}

std::shared_ptr<Image> Image::FlipVertical() const {
    auto output = std::make_shared<Image>();
    output->Prepare(width_, height_, num_of_channels_, bytes_per_channel_);

    vertical_flip_functor func(thrust::raw_pointer_cast(data_.data()), width_, height_,
                               num_of_channels_ * bytes_per_channel_,
                               thrust::raw_pointer_cast(output->data_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(data_.size()), func);
    return output;
}

std::shared_ptr<Image> Image::FlipHorizontal() const {
    auto output = std::make_shared<Image>();
    output->Prepare(width_, height_, num_of_channels_, bytes_per_channel_);

    horizontal_flip_functor func(thrust::raw_pointer_cast(data_.data()), width_, height_,
                                 num_of_channels_ * bytes_per_channel_,
                                 thrust::raw_pointer_cast(output->data_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(data_.size()), func);
    return output;
}

void Image::AllocateDataBuffer() {
    data_.resize(width_ * height_ * num_of_channels_ * bytes_per_channel_);
}