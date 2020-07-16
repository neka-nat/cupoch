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
#include <png.h>

#include "cupoch/geometry/image.h"
#include "cupoch/io/class_io/image_io.h"
#include "cupoch/utility/console.h"

namespace cupoch {

namespace {
using namespace io;

void SetPNGImageFromImage(const geometry::Image &image, png_image &pngimage) {
    pngimage.width = image.width_;
    pngimage.height = image.height_;
    pngimage.format = 0;
    if (image.bytes_per_channel_ == 2) {
        pngimage.format |= PNG_FORMAT_FLAG_LINEAR;
    }
    if (image.num_of_channels_ == 3) {
        pngimage.format |= PNG_FORMAT_FLAG_COLOR;
    }
}

void SetPNGImageFromImage(const HostImage &image, png_image &pngimage) {
    pngimage.width = image.width_;
    pngimage.height = image.height_;
    pngimage.format = 0;
    if (image.bytes_per_channel_ == 2) {
        pngimage.format |= PNG_FORMAT_FLAG_LINEAR;
    }
    if (image.num_of_channels_ == 3) {
        pngimage.format |= PNG_FORMAT_FLAG_COLOR;
    }
}

}  // unnamed namespace

namespace io {

bool ReadImageFromPNG(const std::string &filename, geometry::Image &image) {
    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    if (png_image_begin_read_from_file(&pngimage, filename.c_str()) == 0) {
        utility::LogWarning("Read PNG failed: unable to parse header.");
        return false;
    }

    // We only support two channel types: gray, and RGB.
    // There is no alpha channel.
    // bytes_per_channel is determined by PNG_FORMAT_FLAG_LINEAR flag.
    HostImage host_img;
    host_img.Prepare(pngimage.width, pngimage.height,
                     (pngimage.format & PNG_FORMAT_FLAG_COLOR) ? 3 : 1,
                     (pngimage.format & PNG_FORMAT_FLAG_LINEAR) ? 2 : 1);
    SetPNGImageFromImage(host_img, pngimage);
    if (png_image_finish_read(&pngimage, NULL, host_img.data_.data(), 0,
                              NULL) == 0) {
        utility::LogWarning("Read PNG failed: unable to read file: {}",
                            filename);
        return false;
    }
    host_img.ToDevice(image);
    return true;
}

bool WriteImageToPNG(const std::string &filename,
                     const geometry::Image &image,
                     int quality) {
    if (image.HasData() == false) {
        utility::LogWarning("Write PNG failed: image has no data.");
        return false;
    }
    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    SetPNGImageFromImage(image, pngimage);
    HostImage host_img;
    host_img.FromDevice(image);
    if (png_image_write_to_file(&pngimage, filename.c_str(), 0,
                                host_img.data_.data(), 0, NULL) == 0) {
        utility::LogWarning("Write PNG failed: unable to write file: {}",
                            filename);
        return false;
    }
    return true;
}

bool WriteHostImageToPNG(const std::string &filename,
                         const HostImage &image,
                         int quality) {
    if (image.data_.size() == 0) {
        utility::LogWarning("Write PNG failed: image has no data.");
        return false;
    }
    png_image pngimage;
    memset(&pngimage, 0, sizeof(pngimage));
    pngimage.version = PNG_IMAGE_VERSION;
    SetPNGImageFromImage(image, pngimage);
    if (png_image_write_to_file(&pngimage, filename.c_str(), 0,
                                image.data_.data(), 0, NULL) == 0) {
        utility::LogWarning("Write PNG failed: unable to write file: {}",
                            filename);
        return false;
    }
    return true;
}

}  // namespace io
}  // namespace cupoch