#include "cupoch/geometry/image.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct make_float_image_functor {
    make_float_image_functor(const uint8_t* image, int num_of_channels,
                             int bytes_per_channel,
                             Image::ColorToIntensityConversionType type,
                             uint8_t* fimage)
                             : image_(image), num_of_channels_(num_of_channels),
                               bytes_per_channel_(bytes_per_channel),
                               type_(type), fimage_(fimage) {};
    const uint8_t* image_;
    int num_of_channels_;
    int bytes_per_channel_;
    Image::ColorToIntensityConversionType type_;
    uint8_t* fimage_;
    __device__
    void operator() (size_t idx) {
        float *p = (float *)(fimage_ + idx * 4);
        const uint8_t *pi =
                image_ + idx * num_of_channels_ * bytes_per_channel_;
        if (num_of_channels_ == 1) {
            // grayscale image
            if (bytes_per_channel_ == 1) {
                *p = (float)(*pi) / 255.0f;
            } else if (bytes_per_channel_ == 2) {
                const uint16_t *pi16 = (const uint16_t *)pi;
                *p = (float)(*pi16);
            } else if (bytes_per_channel_ == 4) {
                const float *pf = (const float *)pi;
                *p = *pf;
            }
        } else if (num_of_channels_ == 3) {
            if (bytes_per_channel_ == 1) {
                if (type_ == Image::ColorToIntensityConversionType::Equal) {
                    *p = ((float)(pi[0]) + (float)(pi[1]) + (float)(pi[2])) /
                         3.0f / 255.0f;
                } else if (type_ ==
                           Image::ColorToIntensityConversionType::Weighted) {
                    *p = (0.2990f * (float)(pi[0]) + 0.5870f * (float)(pi[1]) +
                          0.1140f * (float)(pi[2])) /
                         255.0f;
                }
            } else if (bytes_per_channel_ == 2) {
                const uint16_t *pi16 = (const uint16_t *)pi;
                if (type_ == Image::ColorToIntensityConversionType::Equal) {
                    *p = ((float)(pi16[0]) + (float)(pi16[1]) +
                          (float)(pi16[2])) /
                         3.0f;
                } else if (type_ ==
                           Image::ColorToIntensityConversionType::Weighted) {
                    *p = (0.2990f * (float)(pi16[0]) +
                          0.5870f * (float)(pi16[1]) +
                          0.1140f * (float)(pi16[2]));
                }
            } else if (bytes_per_channel_ == 4) {
                const float *pf = (const float *)pi;
                if (type_ == Image::ColorToIntensityConversionType::Equal) {
                    *p = (pf[0] + pf[1] + pf[2]) / 3.0f;
                } else if (type_ ==
                           Image::ColorToIntensityConversionType::Weighted) {
                    *p = (0.2990f * pf[0] + 0.5870f * pf[1] + 0.1140f * pf[2]);
                }
            }
        }
    }
};

}

std::shared_ptr<Image> Image::CreateFloatImage(
        Image::ColorToIntensityConversionType type /* = WEIGHTED*/) const {
    auto fimage = std::make_shared<Image>();
    if (IsEmpty()) {
        return fimage;
    }
    fimage->Prepare(width_, height_, 1, 4);
    make_float_image_functor func(thrust::raw_pointer_cast(data_.data()),
                                  num_of_channels_, bytes_per_channel_, type,
                                  thrust::raw_pointer_cast(fimage->data_.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), 
                     thrust::make_counting_iterator<size_t>(width_ * height_), func);
    return fimage;
}