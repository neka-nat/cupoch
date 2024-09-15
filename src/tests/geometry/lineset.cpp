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
#include "cupoch/geometry/lineset.h"

#include "cupoch/geometry/pointcloud.h"
#include "tests/test_utility/raw.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

TEST(LineSet, Constructor) {
    geometry::LineSet<3> ls;

    // inherited from Geometry2D
    EXPECT_EQ(geometry::Geometry::GeometryType::LineSet, ls.GetGeometryType());
    EXPECT_EQ(3, ls.Dimension());

    // public member variables
    EXPECT_EQ(0u, ls.points_.size());
    EXPECT_EQ(0u, ls.lines_.size());
    EXPECT_EQ(0u, ls.colors_.size());

    // public members
    EXPECT_TRUE(ls.IsEmpty());

    ExpectEQ(Zero3f, ls.GetMinBound());
    ExpectEQ(Zero3f, ls.GetMaxBound());

    EXPECT_FALSE(ls.HasPoints());
    EXPECT_FALSE(ls.HasLines());
    EXPECT_FALSE(ls.HasColors());
}

TEST(LineSet, Clear) {
    int size = 100;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(1000.0, 1000.0, 1000.0);

    Vector2i imin(0, 0);
    Vector2i imax(1000, 1000);

    geometry::LineSet<3> ls;

    std::vector<Eigen::Vector3f> points(size);
    std::vector<Eigen::Vector2i> lines(size);
    std::vector<Eigen::Vector3f> colors(size);
    Rand(points, dmin, dmax, 0);
    Rand(lines, imin, imax, 0);
    Rand(colors, dmin, dmax, 0);
    ls.SetPoints(points);
    ls.SetLines(lines);
    ls.SetColors(colors);

    EXPECT_FALSE(ls.IsEmpty());

    ExpectEQ(Vector3f(19.607843, 0.0, 0.0), ls.GetMinBound());
    ExpectEQ(Vector3f(996.078431, 996.078431, 996.078431), ls.GetMaxBound());

    EXPECT_TRUE(ls.HasPoints());
    EXPECT_TRUE(ls.HasLines());
    EXPECT_TRUE(ls.HasColors());

    ls.Clear();

    // public members
    EXPECT_TRUE(ls.IsEmpty());
    ExpectEQ(Zero3f, ls.GetMinBound());
    ExpectEQ(Zero3f, ls.GetMaxBound());

    EXPECT_FALSE(ls.HasPoints());
    EXPECT_FALSE(ls.HasLines());
    EXPECT_FALSE(ls.HasColors());
}