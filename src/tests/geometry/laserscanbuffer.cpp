#include "cupoch/geometry/laserscanbuffer.h"

#include "tests/test_utility/raw.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

TEST(LaserScanBuffer, Constructor) {
    geometry::LaserScanBuffer scan(100);

    // inherited from Geometry2D
    EXPECT_EQ(geometry::Geometry::GeometryType::LaserScanBuffer, scan.GetGeometryType());
    EXPECT_EQ(3, scan.Dimension());

    // public member variables
    EXPECT_EQ(0u, scan.ranges_.size());
    EXPECT_EQ(0u, scan.intensities_.size());

    // public members
    EXPECT_TRUE(scan.IsEmpty());

    EXPECT_FALSE(scan.HasIntensities());
}

TEST(LaserScanBuffer, AddRanges) {
    geometry::LaserScanBuffer scan(100);
    EXPECT_EQ(scan.num_steps_, 100);
    EXPECT_EQ(scan.bottom_, 0);

    thrust::host_vector<float> hvec(100, 1.0);
    scan.AddRanges(hvec);
    EXPECT_EQ(scan.num_steps_, 100);
    EXPECT_EQ(scan.bottom_, 1);
    EXPECT_EQ(scan.origins_.size(), scan.bottom_);
    EXPECT_EQ(scan.ranges_.size(), scan.num_steps_ * scan.bottom_);
    scan.AddRanges(hvec);
    EXPECT_EQ(scan.num_steps_, 100);
    EXPECT_EQ(scan.bottom_, 2);
    EXPECT_EQ(scan.origins_.size(), scan.bottom_);
    EXPECT_EQ(scan.ranges_.size(), scan.num_steps_ * scan.bottom_);

    scan.Clear();
    scan.AddRanges(hvec, Matrix4f::Identity(), hvec);
    EXPECT_EQ(scan.ranges_.size(), scan.num_steps_ * scan.bottom_);
    EXPECT_EQ(scan.intensities_.size(), scan.num_steps_ * scan.bottom_);
}

TEST(LaserScanBuffer, RangeFilter) {
    geometry::LaserScanBuffer scan(5);
    thrust::host_vector<float> hvec;
    hvec.push_back(1.0);
    hvec.push_back(2.0);
    hvec.push_back(3.0);
    hvec.push_back(4.0);
    hvec.push_back(5.0);

    scan.AddRanges(hvec);
    auto out = scan.RangeFilter(1.5, 3.5);
    auto h_ranges = out->GetRanges();
    EXPECT_TRUE(std::isnan(h_ranges[0]));
    EXPECT_EQ(h_ranges[1], 2.0);
    EXPECT_EQ(h_ranges[2], 3.0);
    EXPECT_TRUE(std::isnan(h_ranges[3]));
    EXPECT_TRUE(std::isnan(h_ranges[4]));
}