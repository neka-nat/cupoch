#include "cupoch/cupoch.h"

int main(int argc, char *argv[]) {
    using namespace cupoch;
    utility::InitializeAllocator();

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 2) {utility::LogInfo("Need an argument of point cloud file name."); return 0;}
    auto pcd = io::CreatePointCloudFromFile(argv[1]);
    pcd->EstimateNormals();
    visualization::DrawGeometries({pcd});
    return 0;
}