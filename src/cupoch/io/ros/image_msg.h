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
#pragma once
#include "cupoch/geometry/image.h"


namespace cupoch {
namespace io {

class ImageMsgInfo {
public:
    ImageMsgInfo(int width, int height,
                 const std::string& encoding,
                 bool is_bigendian,
                 int step)
    : width_(width), height_(height),
    encoding_(encoding), is_bigendian_(is_bigendian),
    step_(step) {};

    ~ImageMsgInfo() {};

    static ImageMsgInfo DefaultCvColor(int width, int height) {
        return ImageMsgInfo(width, height,
                            std::string("bgr8"),
                            false,
                            width * 3);
    };

    int width_;
    int height_;
    std::string encoding_;
    bool is_bigendian_;
    int step_;
};

std::shared_ptr<geometry::Image> CreateFromImageMsg(
    const uint8_t* data, const ImageMsgInfo& info
);

void CreateToImageMsg(uint8_t* data, const ImageMsgInfo& info, const geometry::Image& image);

}
}