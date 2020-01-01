#include "cupoch/io/class_io/pointcloud_io.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/visualization/utility/draw_geometry.h"
#include "cupoch/utility/console.h"

int main(int argc, char *argv[]) {
    using namespace cupoch;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 2) {utility::LogInfo("Need an argument of point cloud file name."); return 0;}
    auto pcd = io::CreatePointCloudFromFile(argv[1]);
    pcd->EstimateNormals();
    visualization::DrawGeometries({pcd});
    return 0;
}