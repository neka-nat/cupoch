#include "cupoch/io/class_io/pointcloud_io.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/registration/registration.h"
#include "cupoch/utility/console.h"

int main(int argc, char *argv[]) {
    using namespace cupoch;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    auto source = io::CreatePointCloudFromFile(argv[1]);
    auto target = io::CreatePointCloudFromFile(argv[2]);
    Eigen::Matrix4f init = (Eigen::Matrix4f() << 0.862, 0.011, -0.507, 0.5,
                                -0.139, 0.967, -0.215, 0.7,
                                0.487, 0.255, 0.835, -1.4,
                                0.0, 0.0, 0.0, 1.0).finished();
    auto res = registration::RegistrationICP(*source, *target, 0.02, init);
    std::cout << res.transformation_ << std::endl;
    return 0;
}