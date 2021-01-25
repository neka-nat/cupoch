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
#include "cupoch/imageproc/sgm.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace imageproc {

SemiGlobalMatching::SemiGlobalMatching(const SGMParameters& params)
: params_(params.p1_,
          params.p2_,
          params.uniqueness_,
          false,
          params.path_type_ == SGMParameters::PathType::ScanPath8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH,
          params.min_disp_,
          params.lr_max_diff_),
sgm_(params.width_,
     params.height_,
     params.disp_size_,
     8,
     8,
     sgm::EXECUTE_INOUT_CUDA2CUDA,
     params_) {}

std::shared_ptr<geometry::Image> SemiGlobalMatching::ProcessFrame(const geometry::Image& left, const geometry::Image& right) {
    auto output = std::make_shared<geometry::Image>();
    if (sgm_.width_ == 0 || sgm_.height_ == 0) {
        utility::LogError("[SemiGlobalMatching::ProcessFrame] Invalid SGM parameters.");
        return output;
    }
    if (left.width_ != right.width_ ||
        left.height_ != right.height_ ||
        left.num_of_channels_ != 1 ||
        right.num_of_channels_ != 1 ||
        left.bytes_per_channel_ != 1 ||
        right.bytes_per_channel_ != 1) {
        utility::LogError("[SemiGlobalMatching::ProcessFrame] Unsupport image type.");
        return output;
    }
    output->Prepare(left.width_, left.height_, 1, 1);
    sgm_.execute(thrust::raw_pointer_cast(left.data_.data()), thrust::raw_pointer_cast(right.data_.data()),
                 thrust::raw_pointer_cast(output->data_.data()));
    return output;
}


}
}