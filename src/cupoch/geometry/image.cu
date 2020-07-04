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
            utility::device_vector<float> g3(3);
            g3[0] = 0.25;
            g3[1] = 0.5;
            g3[2] = 0.25;
            return std::make_pair(g3, g3);
        }
        case Image::FilterType::Gaussian5: {
            utility::device_vector<float> g5(5);
            g5[0] = 0.0625;
            g5[1] = 0.25;
            g5[2] = 0.375;
            g5[3] = 0.25;
            g5[4] = 0.0625;
            return std::make_pair(g5, g5);
        }
        case Image::FilterType::Gaussian7: {
            utility::device_vector<float> g7(7);
            g7[0] = 0.03125;
            g7[1] = 0.109375;
            g7[2] = 0.21875;
            g7[3] = 0.28125;
            g7[4] = 0.21875;
            g7[5] = 0.109375;
            g7[6] = 0.03125;
            return std::make_pair(g7, g7);
        }
        case Image::FilterType::Sobel3Dx: {
            utility::device_vector<float> s31(3);
            utility::device_vector<float> s32(3);
            s31[0] = -1.0;
            s31[1] = 0.0;
            s31[2] = 1.0;
            s32[0] = 1.0;
            s32[1] = 2.0;
            s32[2] = 1.0;
            return std::make_pair(s31, s32);
        }
        case Image::FilterType::Sobel3Dy: {
            utility::device_vector<float> s31(3);
            utility::device_vector<float> s32(3);
            s31[0] = -1.0;
            s31[1] = 0.0;
            s31[2] = 1.0;
            s32[0] = 1.0;
            s32[1] = 2.0;
            s32[2] = 1.0;
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
    clip_intensity_functor(uint8_t *fimage, float min, float max)
        : fimage_(fimage), min_(min), max_(max){};
    uint8_t *fimage_;
    const float min_;
    const float max_;
    __device__ void operator()(size_t idx) {
        float *p = (float *)(fimage_ + idx * sizeof(float));
        *p = max(min(max_, *p), min_);
    }
};

struct linear_transform_functor {
    linear_transform_functor(uint8_t *fimage, float scale, float offset)
        : fimage_(fimage), scale_(scale), offset_(offset){};
    uint8_t *fimage_;
    const float scale_;
    const float offset_;
    __device__ void operator()(size_t idx) {
        float *p = (float *)(fimage_ + idx * sizeof(float));
        (*p) = (float)(scale_ * (*p) + offset_);
    }
};

struct downsample_functor {
    downsample_functor(const uint8_t *src,
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

struct filter_horizontal_functor {
    filter_horizontal_functor(const uint8_t *src,
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
            int x_shift = x + i;
            if (x_shift < 0) x_shift = 0;
            if (x_shift > width_ - 1) x_shift = width_ - 1;
            float *pi =
                    (float *)(src_ + (y * width_ + x_shift) * sizeof(float));
            temp += (*pi * kernel_[i + half_kernel_size_]);
        }
        *po = temp;
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

struct depth_to_float_functor {
    depth_to_float_functor(int depth_scale, int depth_trunc, uint8_t *fimage)
        : depth_scale_(depth_scale),
          depth_trunc_(depth_trunc),
          fimage_(fimage){};
    const int depth_scale_;
    const int depth_trunc_;
    uint8_t *fimage_;
    __device__ void operator()(size_t idx) {
        float *p = (float *)(fimage_ + idx * sizeof(float));
        *p /= (float)depth_scale_;
        if (*p >= depth_trunc_) *p = 0.0f;
    }
};

}  // namespace

Image::Image() : GeometryBase<2>(Geometry::GeometryType::Image) {}
Image::~Image() {}

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

AxisAlignedBoundingBox Image::GetAxisAlignedBoundingBox() const {
    utility::LogError("Image::GetAxisAlignedBoundingBox is not supported");
    return AxisAlignedBoundingBox();
}

Image &Image::Transform(const Eigen::Matrix3f &transformation) {
    utility::LogError("Image::Transform is not supported");
    return *this;
}

Image &Image::Translate(const Eigen::Vector2f &translation, bool relative) {
    utility::LogError("Image::Translate is not supported");
    return *this;
}

Image &Image::Scale(const float scale, bool center) {
    utility::LogError("Image::Scale is not supported");
    return *this;
}

Image &Image::Rotate(const Eigen::Matrix2f &R, bool center) {
    utility::LogError("Image::Rotate is not supported");
    return *this;
}

thrust::host_vector<uint8_t> Image::GetData() const {
    thrust::host_vector<uint8_t> data = data_;
    return data;
}

void Image::SetData(const thrust::host_vector<uint8_t> &data) { data_ = data; }

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
    depth_to_float_functor func(depth_scale, depth_trunc,
                                thrust::raw_pointer_cast(output->data_.data()));
    for_each(thrust::make_counting_iterator<size_t>(0),
             thrust::make_counting_iterator<size_t>(width_ * height_), func);
    return output;
}

Image &Image::ClipIntensity(float min /* = 0.0*/, float max /* = 1.0*/) {
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[ClipIntensity] Unsupported image format.");
    }
    clip_intensity_functor func(thrust::raw_pointer_cast(data_.data()), min,
                                max);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(width_ * height_),
                     func);
    return *this;
}

Image &Image::LinearTransform(float scale, float offset /* = 0.0*/) {
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[LinearTransform] Unsupported image format.");
    }
    linear_transform_functor func(thrust::raw_pointer_cast(data_.data()), scale,
                                  offset);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(width_ * height_),
                     func);
    return *this;
}

std::shared_ptr<Image> Image::Downsample() const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[Downsample] Unsupported image format.");
    }
    int half_width = (int)floor((float)width_ / 2.0);
    int half_height = (int)floor((float)height_ / 2.0);
    output->Prepare(half_width, half_height, 1, 4);

    downsample_functor func(thrust::raw_pointer_cast(data_.data()), width_,
                            thrust::raw_pointer_cast(output->data_.data()),
                            output->width_);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(output->width_ *
                                                            output->height_),
                     func);
    return output;
}

std::shared_ptr<Image> Image::FilterHorizontal(
        const utility::device_vector<float> &kernel) const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4 ||
        kernel.size() % 2 != 1) {
        utility::LogError(
                "[FilterHorizontal] Unsupported image format or kernel "
                "size.");
    }
    output->Prepare(width_, height_, 1, 4);

    const int half_kernel_size = (int)(floor((float)kernel.size() / 2.0));

    filter_horizontal_functor func(
            thrust::raw_pointer_cast(data_.data()), width_,
            thrust::raw_pointer_cast(kernel.data()), half_kernel_size,
            thrust::raw_pointer_cast(output->data_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(width_ * height_),
                     func);
    return output;
}

std::shared_ptr<Image> Image::Filter(Image::FilterType type) const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[Filter] Unsupported image format.");
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
        output.push_back(layer_filtered);
    }
    return output;
}

std::shared_ptr<Image> Image::Filter(
        const utility::device_vector<float> &dx,
        const utility::device_vector<float> &dy) const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError("[Filter] Unsupported image format.");
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

void Image::AllocateDataBuffer() {
    data_.resize(width_ * height_ * num_of_channels_ * bytes_per_channel_);
}