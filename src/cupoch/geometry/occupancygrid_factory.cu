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

#include "cupoch/geometry/occupancygrid.h"
#include "cupoch/geometry/voxelgrid.h"

namespace cupoch {
namespace geometry {

namespace {

struct create_from_voxelgrid_functor {
    create_from_voxelgrid_functor(OccupancyVoxel* voxel,
                                  float voxel_size,
                                  int resolution,
                                  float prob_hit_log)
        : voxel_(voxel),
          voxel_size_(voxel_size),
          resolution_(resolution),
          prob_hit_log_(prob_hit_log){};
    OccupancyVoxel* voxel_;
    float voxel_size_;
    int resolution_;
    float prob_hit_log_;
    __device__ void operator()(const Eigen::Vector3i& key) {
        int h_res = resolution_ / 2;
        if (abs(key[0]) > h_res || abs(key[1]) > h_res || abs(key[2]) > h_res) {
            return;
        }
        const int index = IndexOf(key, voxel_size_);
        voxel_[index] = OccupancyVoxel(key, prob_hit_log_);
    }
};

}  // namespace

std::shared_ptr<OccupancyGrid> OccupancyGrid::CreateFromVoxelGrid(
        const VoxelGrid& input) {
    auto output = std::make_shared<OccupancyGrid>();
    if (input.voxel_size_ <= 0.0) {
        utility::LogError("[CreateFromVoxelGrid] voxel grid  voxel_size <= 0.");
    }
    output->voxel_size_ = input.voxel_size_;
    output->origin_ = input.origin_;
    create_from_voxelgrid_functor func(
            thrust::raw_pointer_cast(output->voxels_.data()),
            output->voxel_size_, output->resolution_, output->prob_hit_log_);

    thrust::for_each(input.voxels_keys_.begin(), input.voxels_keys_.end(),
                     func);
    return output;
}

}  // namespace geometry
}  // namespace cupoch