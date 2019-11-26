#include <gtest/gtest.h>
#include "cupoc/geometry/pointcloud.h"
#include "tests/test_utility/unit_test.h"

using namespace cupoc;
using namespace cupoc::geometry;
using namespace unit_test;

TEST(PointCloud, Constuctor) {
    geometry::PointCloud pc;

    EXPECT_EQ(3, pc.Dimension());

    // public member variables
    EXPECT_EQ(0u, pc.points_.size());
    EXPECT_EQ(0u, pc.normals_.size());
    EXPECT_EQ(0u, pc.colors_.size());

    // public members
    EXPECT_TRUE(pc.IsEmpty());

    ExpectEQ(Zero3f, pc.GetMinBound());
    ExpectEQ(Zero3f, pc.GetMaxBound());

    EXPECT_FALSE(pc.HasPoints());
    EXPECT_FALSE(pc.HasNormals());
    EXPECT_FALSE(pc.HasColors());
}

TEST(PointCloud, Transform) {
    thrust::host_vector<Eigen::Vector3f_u> ref_points;
    ref_points.push_back(Eigen::Vector3f_u(1.411252, 4.274168, 3.130918));
    ref_points.push_back(Eigen::Vector3f_u(1.231757, 4.154505, 3.183678));
    ref_points.push_back(Eigen::Vector3f_u(1.403168, 4.268779, 2.121679));
    ref_points.push_back(Eigen::Vector3f_u(1.456767, 4.304511, 2.640845));
    ref_points.push_back(Eigen::Vector3f_u(1.620902, 4.413935, 1.851255));
    ref_points.push_back(Eigen::Vector3f_u(1.374684, 4.249790, 3.062485));
    ref_points.push_back(Eigen::Vector3f_u(1.328160, 4.218773, 1.795728));
    ref_points.push_back(Eigen::Vector3f_u(1.713446, 4.475631, 1.860145));
    ref_points.push_back(Eigen::Vector3f_u(1.409239, 4.272826, 2.011462));
    ref_points.push_back(Eigen::Vector3f_u(1.480169, 4.320113, 1.177780));

    thrust::host_vector<Eigen::Vector3f_u> ref_normals;
    ref_normals.push_back(Eigen::Vector3f_u(396.470588, 1201.176471, 880.352941));
    ref_normals.push_back(Eigen::Vector3f_u(320.392157, 1081.176471, 829.019608));
    ref_normals.push_back(Eigen::Vector3f_u(268.627451, 817.647059, 406.666667));
    ref_normals.push_back(Eigen::Vector3f_u(338.431373, 1000.392157, 614.117647));
    ref_normals.push_back(Eigen::Vector3f_u(423.137255, 1152.549020, 483.607843));
    ref_normals.push_back(Eigen::Vector3f_u(432.549020, 1337.647059, 964.392157));
    ref_normals.push_back(Eigen::Vector3f_u(139.607843, 443.921569, 189.176471));
    ref_normals.push_back(Eigen::Vector3f_u(291.764706, 762.352941, 317.058824));
    ref_normals.push_back(Eigen::Vector3f_u(134.117647, 407.058824, 191.882353));
    ref_normals.push_back(Eigen::Vector3f_u(274.509804, 801.568627, 218.627451));

    int size = 10;
    geometry::PointCloud pc;

    Eigen::Vector3f vmin(0.0, 0.0, 0.0);
    Eigen::Vector3f vmax(1000.0, 1000.0, 1000.0);

    thrust::host_vector<Eigen::Vector3f_u> points(size);
    Rand(points, vmin, vmax, 0);
    pc.SetPoints(points);

    thrust::host_vector<Eigen::Vector3f_u> normals(size);
    normals.resize(size);
    Rand(normals, vmin, vmax, 0);
    pc.SetNormals(normals);

    Eigen::Matrix4f transformation;
    transformation << 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16;

    pc.Transform(transformation);

    ExpectEQ(ref_points, pc.GetPoints());
    ExpectEQ(ref_normals, pc.GetNormals());
}