#include "cupoch/geometry/occupancygrid.h"

#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

TEST(OccupancyGrid, Bounds) {
    auto occupancy_grid = std::make_shared<geometry::OccupancyGrid>();
    occupancy_grid->origin_ = Eigen::Vector3f(0, 0, 0);
    occupancy_grid->voxel_size_ = 5;
    occupancy_grid->AddVoxel(Eigen::Vector3i(1, 0, 0));
    occupancy_grid->AddVoxel(Eigen::Vector3i(0, 2, 0));
    occupancy_grid->AddVoxel(Eigen::Vector3i(0, 0, 3));
    ExpectEQ(occupancy_grid->GetMinBound(), Eigen::Vector3f(0, 0, 0));
    ExpectEQ(occupancy_grid->GetMaxBound(), Eigen::Vector3f(10, 15, 20));
}

TEST(OccupancyGrid, GetVoxel) {
    auto occupancy_grid = std::make_shared<geometry::OccupancyGrid>();
    occupancy_grid->origin_ = Eigen::Vector3f(0, 0, 0);
    occupancy_grid->voxel_size_ = 1.0;
    occupancy_grid->AddVoxel(Eigen::Vector3i(1, 0, 0), true);
    auto res1 = occupancy_grid->GetVoxel(Eigen::Vector3f(1.5, 0.0, 0.0));
    EXPECT_TRUE(thrust::get<0>(res1));
    EXPECT_FLOAT_EQ(thrust::get<1>(res1).prob_log_, occupancy_grid->prob_hit_log_);
    occupancy_grid->AddVoxel(Eigen::Vector3i(1, 0, 0), true);
    auto res2 = occupancy_grid->GetVoxel(Eigen::Vector3f(1.5, 0.0, 0.0));
    EXPECT_TRUE(thrust::get<0>(res2));
    EXPECT_FLOAT_EQ(thrust::get<1>(res2).prob_log_, 2.0 * occupancy_grid->prob_hit_log_);
    occupancy_grid->AddVoxel(Eigen::Vector3i(1, 0, 0), false);
    auto res3 = occupancy_grid->GetVoxel(Eigen::Vector3f(1.5, 0.0, 0.0));
    EXPECT_TRUE(thrust::get<0>(res3));
    EXPECT_FLOAT_EQ(thrust::get<1>(res3).prob_log_, 2.0 * occupancy_grid->prob_hit_log_ + occupancy_grid->prob_miss_log_);
}

TEST(OccupancyGrid, Insert) {
    auto occupancy_grid = std::make_shared<geometry::OccupancyGrid>();
    occupancy_grid->origin_ = Eigen::Vector3f(-0.5, -0.5, 0);
    occupancy_grid->voxel_size_ = 1.0;
    thrust::host_vector<Eigen::Vector3f> host_points;
    host_points.push_back({0.0, 0.0, 3.5});
    occupancy_grid->Insert(host_points, Eigen::Vector3f::Zero());
    EXPECT_EQ(occupancy_grid->voxels_keys_.size(), 4);
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