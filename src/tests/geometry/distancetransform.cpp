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
    h_keys.push_back(Eigen::Vector3i(5, 5, 5));
    voxelgrid.SetVoxels(h_keys, thrust::host_vector<geometry::Voxel>());
    geometry::DistanceTransform dt(1.0, 512);
    dt.ComputeVoronoiDiagram(voxelgrid);
}