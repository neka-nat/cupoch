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

#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

TEST(OccupancyGrid, Bounds) {
    auto occupancy_grid = std::make_shared<geometry::OccupancyGrid>();
    EXPECT_FLOAT_EQ(occupancy_grid->voxel_size_, 0.05);
    EXPECT_EQ(occupancy_grid->resolution_, 512);
    occupancy_grid->origin_ = Eigen::Vector3f(0, 0, 0);
    occupancy_grid->voxel_size_ = 5;
    occupancy_grid->AddVoxel(Eigen::Vector3i(0, 0, 0));
    occupancy_grid->AddVoxel(Eigen::Vector3i(511, 511, 511));
    Eigen::Vector3f lower = -Eigen::Vector3f::Constant(512 * 5 * 0.5);
    Eigen::Vector3f upper = Eigen::Vector3f::Constant(512 * 5 * 0.5);
    ExpectEQ(occupancy_grid->GetMinBound(), lower);
    ExpectEQ(occupancy_grid->GetMaxBound(), upper);
}

TEST(OccupancyGrid, GetVoxel) {
    auto occupancy_grid = std::make_shared<geometry::OccupancyGrid>();
    occupancy_grid->origin_ = Eigen::Vector3f(0, 0, 0);
    occupancy_grid->voxel_size_ = 1.0;
    size_t h_res = 512 / 2;
    occupancy_grid->AddVoxel(Eigen::Vector3i(h_res + 1, h_res, h_res), true);
    auto res1 = occupancy_grid->GetVoxel(Eigen::Vector3f(1.5, 0.0, 0.0));
    EXPECT_TRUE(thrust::get<0>(res1));
    EXPECT_FLOAT_EQ(thrust::get<1>(res1).prob_log_,
                    occupancy_grid->prob_hit_log_);
    occupancy_grid->AddVoxel(Eigen::Vector3i(h_res + 1, h_res, h_res), true);
    auto res2 = occupancy_grid->GetVoxel(Eigen::Vector3f(1.5, 0.0, 0.0));
    EXPECT_TRUE(thrust::get<0>(res2));
    EXPECT_FLOAT_EQ(thrust::get<1>(res2).prob_log_,
                    2.0 * occupancy_grid->prob_hit_log_);
    occupancy_grid->AddVoxel(Eigen::Vector3i(h_res + 1, h_res, h_res), false);
    auto res3 = occupancy_grid->GetVoxel(Eigen::Vector3f(1.5, 0.0, 0.0));
    EXPECT_TRUE(thrust::get<0>(res3));
    EXPECT_FLOAT_EQ(thrust::get<1>(res3).prob_log_,
                    2.0 * occupancy_grid->prob_hit_log_ +
                            occupancy_grid->prob_miss_log_);
}

TEST(OccupancyGrid, Insert) {
    auto occupancy_grid = std::make_shared<geometry::OccupancyGrid>();
    occupancy_grid->origin_ = Eigen::Vector3f(-0.5, -0.5, 0);
    occupancy_grid->voxel_size_ = 1.0;
    std::vector<Eigen::Vector3f> host_points;
    host_points.push_back({0.0, 0.0, 3.5});
    occupancy_grid->Insert(host_points, Eigen::Vector3f::Zero());
    EXPECT_EQ(occupancy_grid->ExtractKnownVoxels()->size(), 4);
    auto res1 = occupancy_grid->GetVoxel(Eigen::Vector3f(0.0, 0.0, 0.5));
    EXPECT_TRUE(thrust::get<0>(res1));
    auto res2 = occupancy_grid->GetVoxel(Eigen::Vector3f(0.0, 0.0, 1.5));
    EXPECT_TRUE(thrust::get<0>(res2));
    auto res3 = occupancy_grid->GetVoxel(Eigen::Vector3f(0.0, 0.0, 2.5));
    EXPECT_TRUE(thrust::get<0>(res3));
    auto res4 = occupancy_grid->GetVoxel(Eigen::Vector3f(0.0, 0.0, 3.5));
    EXPECT_TRUE(thrust::get<0>(res4));
    auto res5 = occupancy_grid->GetVoxel(Eigen::Vector3f(0.0, 0.0, 4.5));
    EXPECT_FALSE(thrust::get<0>(res5));
}

TEST(OccupancyGrid, SetFreeArea) {
    auto occupancy_grid = std::make_shared<geometry::OccupancyGrid>();
    occupancy_grid->SetFreeArea(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0.1, 0.1, 0.1));
    auto res = occupancy_grid->ExtractFreeVoxels();
    EXPECT_EQ(res->size(), 27);
}