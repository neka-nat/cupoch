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
#include "cupoch/geometry/distancetransform.h"

#include "cupoch/geometry/voxelgrid.h"
#include "tests/test_utility/raw.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

TEST(DistanceTransform, ComputeVoronoiDiagram) {
    geometry::VoxelGrid voxelgrid;
    voxelgrid.voxel_size_ = 1.0;
    std::vector<Eigen::Vector3i> h_keys;
    Eigen::Vector3i ref(5, 5, 5);
    h_keys.push_back(ref);
    voxelgrid.SetVoxels(h_keys, std::vector<geometry::Voxel>());
    geometry::DistanceTransform dt(1.0, 512);
    dt.ComputeVoronoiDiagram(voxelgrid);
    auto v = dt.GetVoxel(Eigen::Vector3f(0.0, 0.0, 0.0));
    EXPECT_TRUE(thrust::get<0>(v));
    EXPECT_EQ(
            thrust::get<1>(v).nearest_index_,
            ref.cast<unsigned short>() + Eigen::Vector3ui16::Constant(512 / 2));
}