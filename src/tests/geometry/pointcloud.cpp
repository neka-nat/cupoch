#include <gtest/gtest.h>
#include "tests/geometry/pointcloud.h"

using namespace cupoc;
using namespace cupoc::geometry;

TEST(PointCloud, Transform) {
    test_pointcloud_transform();
}