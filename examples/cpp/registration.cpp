#include "cupoch/cupoch.h"

int main(int argc, char *argv[]) {
    using namespace cupoch;
    utility::InitializeAllocator();

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    if (argc < 3) {utility::LogInfo("Need two arguments of point cloud file name."); return 0;}

    auto source = std::make_shared<geometry::PointCloud>();
    auto target = std::make_shared<geometry::PointCloud>();
    auto result = std::make_shared<geometry::PointCloud>();
    if (io::ReadPointCloud(argv[1], *source)) {
        utility::LogInfo("Successfully read {}", argv[1]);
    } else {
        utility::LogWarning("Failed to read {}", argv[1]);
    }
    if (io::ReadPointCloud(argv[2], *target)) {
        utility::LogInfo("Successfully read {}", argv[2]);
    } else {
        utility::LogWarning("Failed to read {}", argv[2]);
    }
    Eigen::Matrix4f init = (Eigen::Matrix4f() << 0.862, 0.011, -0.507, 0.5,
                                -0.139, 0.967, -0.215, 0.7,
                                0.487, 0.255, 0.835, -1.4,
                                0.0, 0.0, 0.0, 1.0).finished();
    auto res = registration::RegistrationICP(*source, *target, 0.02, init);
    std::cout << res.transformation_ << std::endl;
    *result = *source;
    result->Transform(res.transformation_);
    visualization::DrawGeometries({source, target, result});
    return 0;
}