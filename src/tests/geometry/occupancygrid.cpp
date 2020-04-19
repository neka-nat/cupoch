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