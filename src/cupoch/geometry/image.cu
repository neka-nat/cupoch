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

struct depth_to_float_functor {
    depth_to_float_functor(int depth_scale, int depth_trunc, uint8_t* fimage)
        : depth_scale_(depth_scale), depth_trunc_(depth_trunc), fimage_(fimage) {};
    const int depth_scale_;
    const int depth_trunc_;
    uint8_t* fimage_;
    __device__
    void operator() (size_t idx) {
        float *p = (float*)(fimage_ + idx * 4);
        *p /= (float)depth_scale_;
        if (*p >= depth_trunc_) *p = 0.0f;
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

std::pair<bool, float> Image::FloatValueAt(float u, float v) const {
    if ((num_of_channels_ != 1) || (bytes_per_channel_ != 4) ||
        (u < 0.0 || u > (float)(width_ - 1) || v < 0.0 ||
         v > (float)(height_ - 1))) {
        return std::make_pair(false, 0.0);
    }
    int ui = std::max(std::min((int)u, width_ - 2), 0);
    int vi = std::max(std::min((int)v, height_ - 2), 0);
    float pu = u - ui;
    float pv = v - vi;
    float value[4] = {*PointerAt<float>(ui, vi), *PointerAt<float>(ui, vi + 1),
                      *PointerAt<float>(ui + 1, vi),
                      *PointerAt<float>(ui + 1, vi + 1)};
    return std::make_pair(true,
                          (value[0] * (1 - pv) + value[1] * pv) * (1 - pu) +
                                  (value[2] * (1 - pv) + value[3] * pv) * pu);
}

template <typename T>
T *Image::PointerAt(int u, int v) const {
    return (T *)(thrust::raw_pointer_cast(data_.data()) + (v * width_ + u) * sizeof(T));
}

template float *Image::PointerAt<float>(int u, int v) const;
template int *Image::PointerAt<int>(int u, int v) const;
template uint8_t *Image::PointerAt<uint8_t>(int u, int v) const;
template uint16_t *Image::PointerAt<uint16_t>(int u, int v) const;

template <typename T>
T *Image::PointerAt(int u, int v, int ch) const {
    return (T *)(thrust::raw_pointer_cast(data_.data()) +
                 ((v * width_ + u) * num_of_channels_ + ch) * sizeof(T));
}

template float *Image::PointerAt<float>(int u, int v, int ch) const;
template int *Image::PointerAt<int>(int u, int v, int ch) const;
template uint8_t *Image::PointerAt<uint8_t>(int u, int v, int ch) const;
template uint16_t *Image::PointerAt<uint16_t>(int u, int v, int ch) const;

std::shared_ptr<Image> Image::ConvertDepthToFloatImage(
        float depth_scale /* = 1000.0*/, float depth_trunc /* = 3.0*/) const {
    // don't need warning message about image type
    // as we call CreateFloatImage
    auto output = CreateFloatImage();
    depth_to_float_functor func(depth_scale, depth_trunc,
                                thrust::raw_pointer_cast(output->data_.data()));
    for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(width_ * height_),
             func);
    return output;
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