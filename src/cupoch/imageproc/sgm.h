/**
 * Copyright (c) 2021 Neka-Nat
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
#define private public
#include <libsgm.h>
#undef private

namespace cupoch {
namespace imageproc {

class SGMOption {
public:
    enum DisparitySizeType {
        DisparitySize64 = 64,
        DisparitySize128 = 128,
        DisparitySize256 = 256,
    };
    enum PathType {
        ScanPath4 = 0,
        ScanPath8 = 1,
    };

    SGMOption(int width = 0,
              int height = 0,
              int p1 = 10,
              int p2 = 120,
              float uniqueness = 0.95f,
              DisparitySizeType disp_size = DisparitySize128,
              PathType path_type = ScanPath8,
              int min_disp = 0,
              int lr_max_diff = 1)
        : width_(width),
          height_(height),
          p1_(p1),
          p2_(p2),
          uniqueness_(uniqueness),
          disp_size_(disp_size),
          path_type_(path_type),
          min_disp_(min_disp),
          lr_max_diff_(lr_max_diff){};
    ~SGMOption(){};
    int width_;
    int height_;
    int p1_;
    int p2_;
    float uniqueness_;
    DisparitySizeType disp_size_;
    PathType path_type_;
    int min_disp_;
    int lr_max_diff_;
};

class SemiGlobalMatching {
public:
    SemiGlobalMatching(const SGMOption& option);
    ~SemiGlobalMatching(){};
    std::shared_ptr<geometry::Image> ProcessFrame(const geometry::Image& left,
                                                  const geometry::Image& right);

private:
    SemiGlobalMatching(const SemiGlobalMatching& other);

public:
    const sgm::StereoSGM::Parameters params_;
    sgm::StereoSGM sgm_;
};

}  // namespace imageproc
}  // namespace cupoch