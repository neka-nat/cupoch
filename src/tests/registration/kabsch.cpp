#include "cupoch/registration/kabsch.h"

#include "cupoch/geometry/pointcloud.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

namespace {
float deg_to_rad(float deg) { return deg / 180.0 * M_PI; }
}  // namespace

TEST(Kabsch, Kabsch) {
    const size_t size = 20;
    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(1000.0, 1000.0, 1000.0);
    thrust::host_vector<Vector3f> points(size);
    Rand(points, vmin, vmax, 0);
    geometry::PointCloud source;
    source.SetPoints(points);
    const float rad = deg_to_rad(30.0f);
    const Matrix4f ref_tf = (Matrix4f() << std::cos(rad), -std::sin(rad), 0.0,
                             0.0, std::sin(rad), std::cos(rad), 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                                    .finished();

    geometry::PointCloud target = source;
    target.Transform(ref_tf);
    const Matrix4f res = registration::Kabsch(source.points_, target.points_);
    std::cout << ref_tf << std::endl;
    std::cout << res << std::endl;
    EXPECT_TRUE(res.isApprox(ref_tf, 1.0e-3));
}