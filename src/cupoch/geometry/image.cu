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
#include <thrust/tabulate.h>
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/image.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

/// Isotropic 2D kernels are separable:
/// two 1D kernels are applied in x and y direction.
std::pair<utility::device_vector<float>, utility::device_vector<float>>
GetFilterKernel(Image::FilterType ftype) {
    switch (ftype) {
        case Image::FilterType::Gaussian3: {
            const float k[3] = {0.25, 0.5, 0.25};
            utility::device_vector<float> g3(k, k + 3);
            return std::make_pair(g3, g3);
        }
        case Image::FilterType::Gaussian5: {
            const float k[5] = {0.0625, 0.25, 0.375, 0.25, 0.0625};
            utility::device_vector<float> g5(k, k + 5);
            return std::make_pair(g5, g5);
        }
        case Image::FilterType::Gaussian7: {
            const float k[7] = {0.03125, 0.109375, 0.21875, 0.28125,
                                0.21875, 0.109375, 0.03125};
            utility::device_vector<float> g7(k, k + 7);
            return std::make_pair(g7, g7);
        }
        case Image::FilterType::Sobel3Dx: {
            const float k1[3] = {-1.0, 0.0, 1.0};
            const float k2[3] = {1.0, 2.0, 1.0};
            utility::device_vector<float> s31(k1, k1 + 3);
            utility::device_vector<float> s32(k2, k2 + 3);
            return std::make_pair(s31, s32);
        }
        case Image::FilterType::Sobel3Dy: {
            const float k1[3] = {-1.0, 0.0, 1.0};
            const float k2[3] = {1.0, 2.0, 1.0};
            utility::device_vector<float> s31(k1, k1 + 3);
            utility::device_vector<float> s32(k2, k2 + 3);
            return std::make_pair(s32, s31);
        }
        default: {
            utility::LogError("[Filter] Unsupported filter type.");
            return std::make_pair(utility::device_vector<float>(),
                                  utility::device_vector<float>());
        }
    }
}

struct transpose_functor {
    transpose_functor(const uint8_t *src,
                      int width,
                      int in_bytes_per_line,
                      int out_bytes_per_line,
                      int bytes_per_pixel,
                      uint8_t *dst)
        : src_(src),
          width_(width),
          in_bytes_per_line_(in_bytes_per_line),
          out_bytes_per_line_(out_bytes_per_line),
          bytes_per_pixel_(bytes_per_pixel),
          dst_(dst){};
    const uint8_t *src_;
    const int width_;
    const int in_bytes_per_line_;
    const int out_bytes_per_line_;
    const int bytes_per_pixel_;
    uint8_t *dst_;
    __device__ void operator()(size_t idx) {
        const int y = idx / width_;
        const int x = idx % width_;
        memcpy(dst_ + x * out_bytes_per_line_ + y * bytes_per_pixel_,
               src_ + y * in_bytes_per_line_ + x * bytes_per_pixel_,
               bytes_per_pixel_ * sizeof(uint8_t));
    }
};

struct clip_intensity_functor {
    clip_intensity_functor(float min, float max) : min_(min), max_(max){};
    const float min_;
    const float max_;
    __device__ void operator()(float &f) { f = max(min(max_, f), min_); }
};

template <typename T>
struct linear_transform_functor {
    linear_transform_functor(float scale, float offset)
        : scale_(scale), offset_(offset){};
    const float scale_;
    const float offset_;
    __device__ void operator()(T &f) { f = (T)(scale_ * (float)f + offset_); }
};

struct downsample_float_functor {
    downsample_float_functor(const uint8_t *src,
                             int src_width,
                             uint8_t *dst,
                             int dst_width)
        : src_(src), src_width_(src_width), dst_(dst), dst_width_(dst_width){};
    const uint8_t *src_;
    const int src_width_;
    uint8_t *dst_;
    const int dst_width_;
    __device__ void operator()(size_t idx) {
        const int y = idx / dst_width_;
        const int x = idx % dst_width_;
        float *p1 =
                (float *)(src_ + (y * 2 * src_width_ + x * 2) * sizeof(float));
        float *p2 = (float *)(src_ +
                              (y * 2 * src_width_ + x * 2 + 1) * sizeof(float));
        float *p3 = (float *)(src_ + ((y * 2 + 1) * src_width_ + x * 2) *
                                             sizeof(float));
        float *p4 = (float *)(src_ + ((y * 2 + 1) * src_width_ + x * 2 + 1) *
                                             sizeof(float));
        float *p = (float *)(dst_ + idx * sizeof(float));
        *p = (*p1 + *p2 + *p3 + *p4) / 4.0f;
    }
};

struct downsample_rgb_functor {
    downsample_rgb_functor(const uint8_t *src,
                           int src_width,
                           int num_of_channels,
                           uint8_t *dst,
                           int dst_width)
        : src_(src),
          src_width_(src_width),
          num_of_channels_(num_of_channels),
          dst_(dst),
          dst_width_(dst_width){};
    const uint8_t *src_;
    const int src_width_;
    const int num_of_channels_;
    uint8_t *dst_;
    const int dst_width_;
    __device__ void operator()(size_t idx) {
        const int y = idx / dst_width_;
        const int x = idx % dst_width_;
        for (int c = 0; c < num_of_channels_; ++c) {
            int p1 = (int)(*(src_ + (y * 2 * src_width_ + x * 2) * 3 + c));
            int p2 = (int)(*(src_ + (y * 2 * src_width_ + x * 2 + 1) * 3 + c));
            int p3 =
                    (int)(*(src_ + ((y * 2 + 1) * src_width_ + x * 2) * 3) + c);
            int p4 = (int)(*(src_ + ((y * 2 + 1) * src_width_ + x * 2 + 1) * 3 +
                             c));
            uint8_t *p = dst_ + idx * 3 + c;
            *p = (uint8_t)((p1 + p2 + p3 + p4) / 4);
        }
    }
};

struct filter_horizontal_float_functor {
    filter_horizontal_float_functor(const uint8_t *src,
                                    int width,
                                    const float *kernel,
                                    int half_kernel_size,
                                    uint8_t *dst)
        : src_(src),
          width_(width),
          kernel_(kernel),
          half_kernel_size_(half_kernel_size),
          dst_(dst){};
    const uint8_t *src_;
    const int width_;
    const float *kernel_;
    const int half_kernel_size_;
    uint8_t *dst_;
    __device__ void operator()(size_t idx) {
        const int y = idx / width_;
        const int x = idx % width_;
        float *po = (float *)(dst_ + idx * sizeof(float));
        float temp = 0;
        for (int i = -half_kernel_size_; i <= half_kernel_size_; i++) {
            int x_shift = min(max(0, x + i), width_ - 1);
            float *pi =
                    (float *)(src_ + (y * width_ + x_shift) * sizeof(float));
            temp += (*pi * kernel_[i + half_kernel_size_]);
        }
        *po = temp;
    }
};

struct filter_horizontal_rgb_functor {
    filter_horizontal_rgb_functor(const uint8_t *src,
                                  int width,
                                  int num_of_channels,
                                  const float *kernel,
                                  int half_kernel_size,
                                  uint8_t *dst)
        : src_(src),
          width_(width),
          num_of_channels_(num_of_channels),
          kernel_(kernel),
          half_kernel_size_(half_kernel_size),
          dst_(dst){};
    const uint8_t *src_;
    const int width_;
    const int num_of_channels_;
    const float *kernel_;
    const int half_kernel_size_;
    uint8_t *dst_;
    __device__ void operator()(size_t idx) {
        const int y = idx / width_;
        const int x = idx % width_;
        for (int c = 0; c < num_of_channels_; ++c) {
            uint8_t *po = dst_ + idx * num_of_channels_ + c;
            float temp = 0;
            for (int i = -half_kernel_size_; i <= half_kernel_size_; i++) {
                int x_shift = min(max(0, x + i), width_ - 1);
                const uint8_t *pi =
                        src_ + (y * width_ + x_shift) * num_of_channels_ + c;
                temp += (*pi * kernel_[i + half_kernel_size_]);
            }
            *po = __float2uint_ru(temp);
        }
    }
};

struct vertical_flip_functor {
    vertical_flip_functor(const uint8_t *src,
                          int width,
                          int height,
                          int bytes_per_pixel,
                          uint8_t *dst)
        : src_(src),
          width_(width),
          height_(height),
          bytes_per_pixel_(bytes_per_pixel),
          dst_(dst){};
    const uint8_t *src_;
    const int width_;
    const int height_;
    const int bytes_per_pixel_;
    uint8_t *dst_;
    __device__ void operator()(size_t idx) {
        const int y = idx / width_;
        const int x = idx % width_;
        memcpy(&dst_[((height_ - y - 1) * width_ + x) * bytes_per_pixel_],
               &src_[idx * bytes_per_pixel_],
               bytes_per_pixel_ * sizeof(uint8_t));
    }
};

struct horizontal_flip_functor {
    horizontal_flip_functor(const uint8_t *src,
                            int width,
                            int bytes_per_pixel,
                            uint8_t *dst)
        : src_(src),
          width_(width),
          bytes_per_pixel_(bytes_per_pixel),
          dst_(dst){};
    const uint8_t *src_;
    const int width_;
    const int bytes_per_pixel_;
    uint8_t *dst_;
    __device__ void operator()(size_t idx) {
        const int y = idx / width_;
        const int x = idx % width_;
        memcpy(&dst_[(y * width_ + (width_ - x - 1)) * bytes_per_pixel_],
               &src_[idx * bytes_per_pixel_],
               bytes_per_pixel_ * sizeof(uint8_t));
    }
};

struct bilateral_filter_functor {
    bilateral_filter_functor(const uint8_t *src,
                             int width,
                             int height,
                             int diameter,
                             float sigma_color,
                             const float *gaussian_const,
                             uint8_t *dst)
        : src_(src),
          width_(width),
          height_(height),
          diameter_(diameter),
          sigma_color_(sigma_color),
          gaussian_const_(gaussian_const),
          dst_(dst){};
    const uint8_t *src_;
    const int width_;
    const int height_;
    const int diameter_;
    const float sigma_color_;
    const float *gaussian_const_;
    uint8_t *dst_;
    __device__ float gaussian(float x, float sig) const {
        return expf(-(x * x) / (2.0f * sig * sig));
    }
    __device__ void operator()(size_t idx) {
        const int y = idx / width_;
        const int x = idx % width_;
        float filtered = 0;
        float total_w = 0;
        const float center_p = *(float *)(src_ + idx * sizeof(float));
        for (int dy = -diameter_; dy <= diameter_; dy++) {
            for (int dx = -diameter_; dx <= diameter_; dx++) {
                const int my = min(max(0, y + dy), height_);
                const int mx = min(max(0, x + dx), width_);
                const float cur_p =
                        *(float *)(src_ + (my * width_ + mx) * sizeof(float));
                const float w = gaussian_const_[dy + diameter_] *
                                gaussian_const_[dx + diameter_] *
                                gaussian(center_p - cur_p, sigma_color_);
                filtered += w * cur_p;
                total_w += w;
            }
        }
        float *p = (float *)(dst_ + idx * sizeof(float));
        *p = filtered / total_w;
    }
};

struct depth_to_float_functor {
    depth_to_float_functor(int depth_scale, int depth_trunc)
        : depth_scale_(depth_scale), depth_trunc_(depth_trunc){};
    const int depth_scale_;
    const int depth_trunc_;
    __device__ void operator()(float &f) {
        f /= (float)depth_scale_;
        if (f >= depth_trunc_) f = 0.0f;
    }
};

}  // namespace

Image::Image() : GeometryBaseNoTrans2D(Geometry::GeometryType::Image) {}
Image::~Image() {}
Image::Image(const Image &other)
    : GeometryBaseNoTrans2D(Geometry::GeometryType::Image),
      width_(other.width_),
      height_(other.height_),
      num_of_channels_(other.num_of_channels_),
      bytes_per_channel_(other.bytes_per_channel_),
      data_(other.data_) {}

Image &Image::operator=(const Image &other) {
    width_ = other.width_;
    height_ = other.height_;
    num_of_channels_ = other.num_of_channels_;
    bytes_per_channel_ = other.bytes_per_channel_;
    data_ = other.data_;
    return *this;
}

Image &Image::Clear() {
    width_ = 0;
    height_ = 0;
    num_of_channels_ = 0;
    bytes_per_channel_ = 0;
    data_.clear();
    return *this;
}

bool Image::IsEmpty() const { return !HasData(); }

Eigen::Vector2f Image::GetMinBound() const { return Eigen::Vector2f(0.0, 0.0); }

Eigen::Vector2f Image::GetMaxBound() const {
    return Eigen::Vector2f(width_, height_);
}

Eigen::Vector2f Image::GetCenter() const {
    return Eigen::Vector2f(width_ / 2, height_ / 2);
}

std::vector<uint8_t> Image::GetData() const {
    std::vector<uint8_t> data(data_.size());
    copy_device_to_host(data_, data);
    return data;
}

void Image::SetData(const thrust::host_vector<uint8_t> &data) { data_ = data; }

void Image::SetData(const std::vector<uint8_t> &data) {
    data_.resize(data.size());
    copy_host_to_device(data, data_);
}

bool Image::TestImageBoundary(float u,
                              float v,
                              float inner_margin /* = 0.0 */) const {
    return (u >= inner_margin && u < width_ - inner_margin &&
            v >= inner_margin && v < height_ - inner_margin);
}

std::pair<bool, float> Image::FloatValueAt(float u, float v) const {
    auto output = geometry::FloatValueAt(thrust::raw_pointer_cast(data_.data()),
                                         u, v, width_, height_,
                                         num_of_channels_, bytes_per_channel_);
    return std::make_pair(output.first, output.second);
}

std::shared_ptr<Image> Image::ConvertDepthToFloatImage(
        float depth_scale /* = 1000.0*/, float depth_trunc /* = 3.0*/) const {
    // don't need warning message about image type
    // as we call CreateFloatImage
    auto output = CreateFloatImage();
    depth_to_float_functor func(depth_scale, depth_trunc);
    float *pt = (float *)thrust::raw_pointer_cast(output->data_.data());
    for_each(thrust::device, pt, pt + (width_ * height_), func);
    return output;
}

Image &Image::ClipIntensity(float min /* = 0.0*/, float max /* = 1.0*/) {
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[ClipIntensity] Unsupported image format.");
        return *this;
    }
    clip_intensity_functor func(min, max);
    float *pt = (float *)thrust::raw_pointer_cast(data_.data());
    thrust::for_each(thrust::device, pt, pt + (width_ * height_), func);
    return *this;
}

Image &Image::LinearTransform(float scale, float offset /* = 0.0*/) {
    if (bytes_per_channel_ != 1 &&
        (num_of_channels_ != 1 || bytes_per_channel_ != 4)) {
        utility::LogError("[LinearTransform] Unsupported image format.");
        return *this;
    }
    if (bytes_per_channel_ == 1) {
        linear_transform_functor<uint8_t> func(scale, offset);
        uint8_t *pt = thrust::raw_pointer_cast(data_.data());
        thrust::for_each(thrust::device, pt,
                         pt + (width_ * height_ * num_of_channels_), func);
    } else if (bytes_per_channel_ == 4) {
        linear_transform_functor<float> func(scale, offset);
        float *pt = (float *)thrust::raw_pointer_cast(data_.data());
        thrust::for_each(thrust::device, pt, pt + (width_ * height_), func);
    }
    return *this;
}

std::shared_ptr<Image> Image::Downsample() const {
    auto output = std::make_shared<Image>();
    if ((num_of_channels_ != 1 || bytes_per_channel_ != 4) &&
        (num_of_channels_ != 3 || bytes_per_channel_ != 1)) {
        utility::LogError("[Downsample] Unsupported image format.");
        return output;
    }
    int half_width = (int)floor((float)width_ / 2.0);
    int half_height = (int)floor((float)height_ / 2.0);
    output->Prepare(half_width, half_height, num_of_channels_,
                    bytes_per_channel_);

    if (num_of_channels_ == 1) {
        downsample_float_functor func(
                thrust::raw_pointer_cast(data_.data()), width_,
                thrust::raw_pointer_cast(output->data_.data()), output->width_);
        thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(
                                 output->width_ * output->height_),
                         func);
    } else {
        downsample_rgb_functor func(
                thrust::raw_pointer_cast(data_.data()), width_,
                num_of_channels_,
                thrust::raw_pointer_cast(output->data_.data()), output->width_);
        thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(
                                 output->width_ * output->height_),
                         func);
    }
    return output;
}

std::shared_ptr<Image> Image::FilterHorizontal(
        const utility::device_vector<float> &kernel) const {
    auto output = std::make_shared<Image>();
    if ((num_of_channels_ != 1 || bytes_per_channel_ != 4) &&
                (num_of_channels_ != 3 || bytes_per_channel_ != 1) ||
        kernel.size() % 2 != 1) {
        utility::LogError(
                "[FilterHorizontal] Unsupported image format or kernel "
                "size.");
    }
    output->Prepare(width_, height_, 1, 4);

    const int half_kernel_size = (int)(floor((float)kernel.size() / 2.0));

    if (num_of_channels_ == 1) {
        filter_horizontal_float_functor func(
                thrust::raw_pointer_cast(data_.data()), width_,
                thrust::raw_pointer_cast(kernel.data()), half_kernel_size,
                thrust::raw_pointer_cast(output->data_.data()));
        thrust::for_each(
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(width_ * height_), func);
    } else {
        filter_horizontal_rgb_functor func(
                thrust::raw_pointer_cast(data_.data()), width_,
                num_of_channels_, thrust::raw_pointer_cast(kernel.data()),
                half_kernel_size,
                thrust::raw_pointer_cast(output->data_.data()));
        thrust::for_each(
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(width_ * height_), func);
    }
    return output;
}

std::shared_ptr<Image> Image::FilterHorizontal(
        const std::vector<float> &kernel) const {
    utility::device_vector<float> d_kernel(kernel.size());
    copy_host_to_device(kernel, d_kernel);
    return FilterHorizontal(d_kernel);
}

std::shared_ptr<Image> Image::Filter(Image::FilterType type) const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[Filter] Unsupported image format.");
        return output;
    }

    auto kernels = GetFilterKernel(type);
    output = Filter(kernels.first, kernels.second);
    return output;
}

ImagePyramid Image::FilterPyramid(const ImagePyramid &input,
                                  Image::FilterType type) {
    std::vector<std::shared_ptr<Image>> output;
    for (size_t i = 0; i < input.size(); i++) {
        auto layer_filtered = input[i]->Filter(type);
        output.emplace_back(layer_filtered);
    }
    return output;
}

ImagePyramid Image::BilateralFilterPyramid(const ImagePyramid &input,
                                           int diameter,
                                           float sigma_color,
                                           float sigma_space) {
    std::vector<std::shared_ptr<Image>> output;
    for (size_t i = 0; i < input.size(); i++) {
        auto layer_filtered =
                input[i]->BilateralFilter(diameter, sigma_color, sigma_space);
        output.emplace_back(layer_filtered);
    }
    return output;
}

std::shared_ptr<Image> Image::Filter(
        const utility::device_vector<float> &dx,
        const utility::device_vector<float> &dy) const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[Filter] Unsupported image format.");
        return output;
    }

    auto temp1 = FilterHorizontal(dx);
    auto temp2 = temp1->Transpose();
    auto temp3 = temp2->FilterHorizontal(dy);
    auto temp4 = temp3->Transpose();
    return temp4;
}

std::shared_ptr<Image> Image::Transpose() const {
    auto output = std::make_shared<Image>();
    output->Prepare(height_, width_, num_of_channels_, bytes_per_channel_);

    int out_bytes_per_line = output->BytesPerLine();
    int in_bytes_per_line = BytesPerLine();
    int bytes_per_pixel = num_of_channels_ * bytes_per_channel_;

    transpose_functor func(thrust::raw_pointer_cast(data_.data()), width_,
                           in_bytes_per_line, out_bytes_per_line,
                           bytes_per_pixel,
                           thrust::raw_pointer_cast(output->data_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(width_ * height_),
                     func);
    return output;
}

std::shared_ptr<Image> Image::FlipVertical() const {
    auto output = std::make_shared<Image>();
    output->Prepare(width_, height_, num_of_channels_, bytes_per_channel_);

    vertical_flip_functor func(thrust::raw_pointer_cast(data_.data()), width_,
                               height_, num_of_channels_ * bytes_per_channel_,
                               thrust::raw_pointer_cast(output->data_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(width_ * height_),
                     func);
    return output;
}

std::shared_ptr<Image> Image::FlipHorizontal() const {
    auto output = std::make_shared<Image>();
    output->Prepare(width_, height_, num_of_channels_, bytes_per_channel_);

    horizontal_flip_functor func(
            thrust::raw_pointer_cast(data_.data()), width_,
            num_of_channels_ * bytes_per_channel_,
            thrust::raw_pointer_cast(output->data_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(width_ * height_),
                     func);
    return output;
}

std::shared_ptr<Image> Image::BilateralFilter(int diameter,
                                              float sigma_color,
                                              float sigma_space) const {
    auto output = std::make_shared<Image>();
    if (diameter >= 64) {
        utility::LogError("[BilateralFilter] Diameter should be less than 64.");
        return output;
    }
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[BilateralFilter] Unsupported image format.");
        return output;
    }
    output->Prepare(width_, height_, num_of_channels_, bytes_per_channel_);
    float fgaussian[64];
    const float sigma2 = sigma_space * sigma_space;
    for (int i = 0; i < 2 * diameter + 1; i++) {
        const float x = i - diameter;
        fgaussian[i] = std::exp(-(x * x) / (2 * sigma2));
    }
    utility::device_vector<float> gaussian_const(fgaussian, fgaussian + 64);
    bilateral_filter_functor func(
            thrust::raw_pointer_cast(data_.data()), width_, height_, diameter,
            sigma_color, thrust::raw_pointer_cast(gaussian_const.data()),
            thrust::raw_pointer_cast(output->data_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(width_ * height_),
                     func);
    return output;
}

void Image::AllocateDataBuffer() {
    data_.resize(width_ * height_ * num_of_channels_ * bytes_per_channel_);
}