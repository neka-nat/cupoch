#include "cupoch/integration/uniform_tsdfvolume.h"
#include "tests/test_utility/unit_test.h"

using namespace cupoch;
using namespace unit_test;

TEST(UniformTSDFVolume, Constructor) {
    float length = 4.0;
    int resolution = 128;
    float sdf_trunc = 0.04;
    auto color_type = integration::TSDFVolumeColorType::RGB8;
    integration::UniformTSDFVolume tsdf_volume(
            length, resolution, sdf_trunc,
            integration::TSDFVolumeColorType::RGB8);

    // TSDFVolume base class attributes
    EXPECT_EQ(tsdf_volume.voxel_length_, length / resolution);
    EXPECT_EQ(tsdf_volume.sdf_trunc_, sdf_trunc);
    EXPECT_EQ(tsdf_volume.color_type_, color_type);

    // UniformTSDFVolume attributes
    ExpectEQ(tsdf_volume.origin_, Eigen::Vector3f(0, 0, 0));
    EXPECT_EQ(tsdf_volume.length_, length);
    EXPECT_EQ(tsdf_volume.resolution_, resolution);
    EXPECT_EQ(tsdf_volume.voxel_num_, resolution * resolution * resolution);
    EXPECT_EQ(int(tsdf_volume.voxels_.size()), tsdf_volume.voxel_num_);
}