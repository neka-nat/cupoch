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
#include "cupoch/geometry/image.h"
#include "cupoch/io/class_io/image_io.h"
#include "cupoch/utility/helper.h"
#include "cupoch/utility/platform.h"

using namespace cupoch;
using namespace cupoch::io;

void HostImage::FromDevice(const geometry::Image& image) {
    data_.resize(image.data_.size());
    Prepare(image.width_, image.height_, image.num_of_channels_,
            image.bytes_per_channel_);
    copy_device_to_host(image.data_, data_);
}

void HostImage::ToDevice(geometry::Image& image) const {
    image.Prepare(width_, height_, num_of_channels_, bytes_per_channel_);
    image.data_.resize(data_.size());
    copy_host_to_device(data_, image.data_);
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