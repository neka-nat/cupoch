#include "cupoch/collision/collision.h"
#include "cupoch/geometry/voxelgrid.h"

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