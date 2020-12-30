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
#include "cupoch/io/ros/image_msg.h"
#include "cupoch/utility/platform.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace io {

namespace {

struct reverse_color_oder_functor {
    reverse_color_oder_functor(uint8_t* data, int width)
    : data_(data), width_(width) {}
    uint8_t* data_;
    int width_;
    __device__ __host__
    void operator() (size_t idx) {
        thrust::swap(data_[3 * idx], data_[3 * idx + 2]);
    }
};

}

std::shared_ptr<geometry::Image> CreateFromImageMsg(
        const uint8_t* data, const ImageMsgInfo& info
) {
    auto out = std::make_shared<geometry::Image>();
    int total_size = info.height_ * info.step_;
    if (info.encoding_ == "bgr8") {
        out->Prepare(info.width_, info.height_, 3, 1);
        cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(out->data_.data()), data, total_size, cudaMemcpyHostToDevice));
        reverse_color_oder_functor func(thrust::raw_pointer_cast(out->data_.data()), out->width_);
        thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(info.width_ * info.height_),
                         func);
        return out;
    } else if (info.encoding_ == "rgb8") {
        out->Prepare(info.width_, info.height_, 3, 1);
        cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(out->data_.data()), data, total_size, cudaMemcpyHostToDevice));
        return out;
    } else {
        utility::LogError("[CreateFromImageMsg] Unsupport encoding type.");
        return out;
    }
}

void CreateToImageMsg(uint8_t* data, const ImageMsgInfo& info, const geometry::Image& image) {
    if (!image.HasData()) {
        return;
    }
    if (info.encoding_ == "bgr8") {
        utility::device_vector<uint8_t> dv_data = image.data_;
        reverse_color_oder_functor func(thrust::raw_pointer_cast(dv_data.data()), image.width_);
        thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator<size_t>(image.width_ * image.height_),
                         func);
        cudaSafeCall(cudaMemcpy(data, thrust::raw_pointer_cast(dv_data.data()), dv_data.size(), cudaMemcpyDeviceToHost));
    } else if (info.encoding_ == "rgb8") {
        cudaSafeCall(cudaMemcpy(data, thrust::raw_pointer_cast(image.data_.data()), image.data_.size(), cudaMemcpyDeviceToHost));
    } else {
        utility::LogError("[CreateToImageMsg] Unsupport encoding type.");
        return;
    }
}

}
}