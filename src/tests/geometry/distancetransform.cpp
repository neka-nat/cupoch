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
    thrust::host_vector<Eigen::Vector3i> h_keys;
    Eigen::Vector3i ref(5, 5, 5);
    h_keys.push_back(ref);
    voxelgrid.SetVoxels(h_keys, thrust::host_vector<geometry::Voxel>());
    geometry::DistanceTransform dt(1.0, 512);
    dt.ComputeVoronoiDiagram(voxelgrid);
    auto v = dt.GetVoxel(Eigen::Vector3f(0.0, 0.0, 0.0));
    EXPECT_TRUE(thrust::get<0>(v));
    EXPECT_EQ(thrust::get<1>(v).nearest_index_, ref.cast<unsigned short>() + Eigen::Vector3ui16::Constant(512 / 2));
}