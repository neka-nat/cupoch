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
#include "cupoch/collision/collision.h"

#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/geometry/lineset.h"
#include "tests/test_utility/raw.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

TEST(Collision, VoxelVoxel) {
    geometry::VoxelGrid voxel1;
    geometry::VoxelGrid voxel2;
    auto res1 = collision::ComputeIntersection(voxel1, voxel2);
    EXPECT_FALSE(res1->IsCollided());
    voxel1.voxel_size_ = 1.0;
    voxel2.voxel_size_ = 1.0;
    voxel1.AddVoxel(geometry::Voxel(Eigen::Vector3i(0, 0, 0)));
    voxel2.AddVoxel(geometry::Voxel(Eigen::Vector3i(0, 0, 0)));
    auto res2 = collision::ComputeIntersection(voxel1, voxel2);
    EXPECT_TRUE(res2->IsCollided());
    voxel1.AddVoxel(geometry::Voxel(Eigen::Vector3i(5, 0, 0)));
    voxel2.AddVoxel(geometry::Voxel(Eigen::Vector3i(10, 0, 0)));
    auto res3 = collision::ComputeIntersection(voxel1, voxel2);
    EXPECT_EQ(res3->collision_index_pairs_.size(), 1);
    EXPECT_EQ(res3->GetCollisionIndexPairs()[0], Eigen::Vector2i(0, 0));
}

TEST(Collision, VoxelLineSet) {
    geometry::VoxelGrid voxel;
    geometry::LineSet<3> lineset;
    auto res1 = collision::ComputeIntersection(voxel, lineset);
    EXPECT_FALSE(res1->IsCollided());
    voxel.voxel_size_ = 1.0;
    voxel.AddVoxel(geometry::Voxel(Eigen::Vector3i(0, 0, 0)));
    voxel.AddVoxel(geometry::Voxel(Eigen::Vector3i(5, 0, 0)));
    std::vector<Eigen::Vector3f> points = {
        Eigen::Vector3f(-0.1, -0.1, -0.1),
        Eigen::Vector3f(-0.1, -0.1, 1.1),
        Eigen::Vector3f(1.1, 1.1, 1.1)
        };
    std::vector<Eigen::Vector2i> lines = {
        Eigen::Vector2i(0, 1),
        Eigen::Vector2i(0, 2)
    };
    lineset.SetPoints(points);
    lineset.SetLines(lines);

    auto res2 = collision::ComputeIntersection(voxel, lineset);
    EXPECT_TRUE(res2->IsCollided());
    EXPECT_EQ(res2->collision_index_pairs_.size(), 1);
    EXPECT_EQ(res2->GetCollisionIndexPairs()[0], Eigen::Vector2i(0, 1));
}