/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
**/
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
    unsigned int buffer_size = 10;
    geometry::LaserScanBuffer scan(100, buffer_size);
    EXPECT_EQ(scan.num_steps_, 100);
    EXPECT_EQ(scan.bottom_, 0);

    std::vector<float> hvec(100, 1.0);
    scan.AddRanges(hvec);
    EXPECT_EQ(scan.num_steps_, 100);
    EXPECT_EQ(scan.bottom_, 1);
    EXPECT_EQ(scan.origins_.size(), scan.bottom_);
    EXPECT_EQ(scan.ranges_.size(), scan.num_steps_ * scan.bottom_);
    EXPECT_EQ(scan.GetNumScans(), 1);
    scan.AddRanges(hvec);
    EXPECT_EQ(scan.num_steps_, 100);
    EXPECT_EQ(scan.bottom_, 2);
    EXPECT_EQ(scan.origins_.size(), scan.bottom_);
    EXPECT_EQ(scan.ranges_.size(), scan.num_steps_ * scan.bottom_);
    EXPECT_EQ(scan.GetNumScans(), 2);
    auto latest = scan.PopOneScan();
    EXPECT_EQ(latest->num_steps_, 100);
    EXPECT_EQ(scan.GetNumScans(), 1);

    scan.Clear();
    for (int i = 0; i < buffer_size; ++i) {
        EXPECT_FALSE(scan.IsFull());
        scan.AddRanges(hvec, Matrix4f::Identity(), hvec);
        EXPECT_EQ(scan.ranges_.size(), scan.num_steps_ * scan.GetNumScans());
        EXPECT_EQ(scan.intensities_.size(), scan.num_steps_ * scan.GetNumScans());
        EXPECT_EQ(scan.GetNumScans(), i + 1);
    }
    EXPECT_TRUE(scan.IsFull());
}

TEST(LaserScanBuffer, RangeFilter) {
    geometry::LaserScanBuffer scan(5);
    std::vector<float> hvec;
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