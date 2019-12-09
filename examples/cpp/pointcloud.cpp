#include "cupoch/io/class_io/pointcloud_io.h"
#include "cupoch/utility/console.h"

int main(int argc, char *argv[]) {
    using namespace cupoch;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    auto pcd = io::CreatePointCloudFromFile(argv[1]);
    pcd->EstimateNormals();
    return 0;
}