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

    thrust::host_vector<Eigen::Vector3f> points(size);
    thrust::host_vector<Eigen::Vector2i> lines(size);
    thrust::host_vector<Eigen::Vector3f> colors(size);
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