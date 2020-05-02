#include "cupoch/integration/uniform_tsdfvolume.h"
#include "cupoch/io/class_io/image_io.h"
#include "cupoch/utility/filesystem.h"
#include "tests/test_utility/unit_test.h"

using namespace cupoch;
using namespace unit_test;

bool ReadPoses(const std::string& trajectory_path,
               thrust::host_vector<Eigen::Matrix4f>& poses) {
    FILE* f = utility::filesystem::FOpen(trajectory_path, "r");
    if (f == NULL) {
        utility::LogWarning("Read poses failed: unable to open file: {}",
                            trajectory_path);
        return false;
    }
    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    Eigen::Matrix4f pose;

    auto read_pose = [&pose, &line_buffer, f]() -> bool {
        // Read meta line
        if (!fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
            return false;
        }
        // Read 4x4 matrix
        for (size_t row = 0; row < 4; ++row) {
            if (!fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
                return false;
            }
            if (sscanf(line_buffer, "%f %f %f %f", &pose(row, 0),
                       &pose(row, 1), &pose(row, 2), &pose(row, 3)) != 4) {
                return false;
            }
        }
        return true;
    };

    while (read_pose()) {
        // Copy to poses
        poses.push_back(pose);
    }

    fclose(f);
    return true;
}

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

TEST(UniformTSDFVolume, RealData) {
    std::string test_data_dir = std::string(TEST_DATA_DIR);

    // Poses
    std::string trajectory_path = test_data_dir + "/rgbd/odometry.log";
    thrust::host_vector<Eigen::Matrix4f> poses;
    if (!ReadPoses(trajectory_path, poses)) {
        throw std::runtime_error("Cannot read trajectory file");
    }

    // Extrinsics
    thrust::host_vector<Eigen::Matrix4f> extrinsics;
    for (const auto& pose : poses) {
        extrinsics.push_back(pose.inverse());
    }

    // Intrinsics
    camera::PinholeCameraIntrinsic intrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);

    // TSDF init
    integration::UniformTSDFVolume tsdf_volume(
            4.0, 100, 0.04, integration::TSDFVolumeColorType::RGB8);

    // Integrate RGBD frames
    for (size_t i = 0; i < poses.size(); ++i) {
        // Color
        geometry::Image im_color;
        std::ostringstream im_color_path;
        im_color_path << TEST_DATA_DIR << "/rgbd/color/" << std::setfill('0')
                      << std::setw(5) << i << ".jpg";
        io::ReadImage(im_color_path.str(), im_color);

        // Depth
        geometry::Image im_depth;
        std::ostringstream im_depth_path;
        im_depth_path << TEST_DATA_DIR << "/rgbd/depth/" << std::setfill('0')
                      << std::setw(5) << i << ".png";
        io::ReadImage(im_depth_path.str(), im_depth);

        // Ingegrate
        std::shared_ptr<geometry::RGBDImage> im_rgbd =
                geometry::RGBDImage::CreateFromColorAndDepth(
                        im_color, im_depth, /*depth_scale*/ 1000.0,
                        /*depth_func*/ 4.0, /*convert_rgb_to_intensity*/ false);
        tsdf_volume.Integrate(*im_rgbd, intrinsic, extrinsics[i]);
    }

    // These hard-coded values are for unit test only. They are used to make
    // sure that after code refactoring, the numerical values still stay the
    // same. However, using different parameters or algorithmtic improvements
    // could invalidate these reference values. We use a custom threshold 0.1
    // to account for acccumulative floating point errors.

    // Extract mesh
    // std::shared_ptr<geometry::TriangleMesh> mesh =
    //         tsdf_volume.ExtractTriangleMesh();
    // EXPECT_EQ(mesh->vertices_.size(), 3198u);
    // EXPECT_EQ(mesh->triangles_.size(), 4402u);
    Eigen::Vector3f color_sum(0, 0, 0);
    // for (const Eigen::Vector3f& color : mesh->GetVertexColors()) {
    //     color_sum += color;
    // }
    // ExpectEQ(color_sum, Eigen::Vector3f(2703.841944, 2561.480949, 2481.503805),
    //          /*threshold*/ 0.1);

    // Extract point cloud
    std::shared_ptr<geometry::PointCloud> pcd = tsdf_volume.ExtractPointCloud();
    EXPECT_EQ(pcd->points_.size(), 2227u);
    EXPECT_EQ(pcd->colors_.size(), 2227u);
    color_sum << 0, 0, 0;
    for (const Eigen::Vector3f& color : pcd->GetColors()) {
        color_sum += color;
    }
    ExpectEQ(color_sum, Eigen::Vector3f(1877.673116, 1862.126057, 1862.190616),
             /*threshold*/ 0.1);
    Eigen::Vector3f normal_sum(0, 0, 0);
    for (const Eigen::Vector3f& normal : pcd->GetNormals()) {
        normal_sum += normal;
    }
    ExpectEQ(normal_sum, Eigen::Vector3f(-161.569098, -95.969433, -1783.167177),
             /*threshold*/ 0.1);

    // Extract voxel cloud
    std::shared_ptr<geometry::PointCloud> voxel_pcd =
            tsdf_volume.ExtractVoxelPointCloud();
    EXPECT_EQ(voxel_pcd->points_.size(), 4488u);
    EXPECT_EQ(voxel_pcd->colors_.size(), 4488u);
    color_sum << 0, 0, 0;
    for (const Eigen::Vector3f& color : voxel_pcd->GetColors()) {
        color_sum += color;
    }
    ExpectEQ(color_sum, Eigen::Vector3f(2096.428416, 2096.428416, 2096.428416),
             /*threshold*/ 0.1);
}