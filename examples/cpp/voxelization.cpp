#include "cupoch/cupoch.h"

using namespace cupoch;

void PrintVoxelGridInformation(const geometry::VoxelGrid& voxel_grid) {
    utility::LogInfo("geometry::VoxelGrid with {:d} voxels",
                     voxel_grid.voxels_keys_.size());
    utility::LogInfo("               origin: [{:f} {:f} {:f}]",
                     voxel_grid.origin_(0), voxel_grid.origin_(1),
                     voxel_grid.origin_(2));
    utility::LogInfo("               voxel_size: {:f}", voxel_grid.voxel_size_);
    return;
}

int main(int argc, char** args) {
    using namespace cupoch;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    if (argc < 3) {
        PrintCupochVersion();
        // clang-format off
        utility::LogInfo("Usage:");
        utility::LogInfo("    > Voxelization [pointcloud_filename] [voxel_filename_ply]");
        // clang-format on
        return 1;
    }

    auto pcd = io::CreatePointCloudFromFile(args[1]);
    auto voxel = geometry::VoxelGrid::CreateFromPointCloud(*pcd, 0.05);
    PrintVoxelGridInformation(*voxel);
    visualization::DrawGeometries({pcd, voxel});
    io::WriteVoxelGrid(args[2], *voxel, true);

    auto voxel_read = io::CreateVoxelGridFromFile(args[2]);
    PrintVoxelGridInformation(*voxel_read);
    visualization::DrawGeometries({pcd, voxel_read});
}