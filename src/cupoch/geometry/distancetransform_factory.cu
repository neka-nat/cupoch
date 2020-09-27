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
#include "cupoch/geometry/distancetransform.h"

namespace cupoch {
namespace geometry {

std::shared_ptr<DistanceTransform> DistanceTransform::CreateFromOccupancyGrid(const OccupancyGrid &input) {
    auto output = std::make_shared<DistanceTransform>();
    if (input.voxel_size_ <= 0.0) {
        utility::LogError("[CreateOccupancyGrid] occupancy grid  voxel_size <= 0.");
    }
    output->voxel_size_ = input.voxel_size_;
    output->origin_ = input.origin_;
    output->resolution_ = input.resolution_;
    output->voxels_.resize(input.voxels_.size());
    std::shared_ptr<utility::device_vector<OccupancyVoxel>> occvoxels = input.ExtractOccupiedVoxels();
    thrust::transform(occvoxels->begin(), occvoxels->end(),
                      thrust::make_permutation_iterator(
                            output->voxels_.begin(),
                            thrust::make_transform_iterator(occvoxels->begin(),
                                                            [res = output->resolution_] __device__ (const OccupancyVoxel& voxel) {
                                                                return IndexOf(voxel.grid_index_.cast<int>(), res);
                                                            })),
                      [res = output->resolution_] __device__ (const OccupancyVoxel& voxel) {
                          return DistanceVoxel(voxel.grid_index_, 0);
                      });
    return output;
}

}
}
