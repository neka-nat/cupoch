#include <gtest/gtest.h>
#include "cupoch/geometry/pointcloud.h"
#include "tests/test_utility/unit_test.h"
#include <thrust/unique.h>

using namespace Eigen;
using namespace cupoch;
using namespace cupoch::geometry;
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

TEST(PointCloud, Clear) {
    int size = 100;

    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(1000.0, 1000.0, 1000.0);

    geometry::PointCloud pc;

    thrust::host_vector<Vector3f_u> points(size);
    Rand(points, vmin, vmax, 0);
    thrust::host_vector<Vector3f_u> normals(size);
    Rand(normals, vmin, vmax, 0);
    thrust::host_vector<Vector3f_u> colors(size);
    Rand(colors, vmin, vmax, 0);
    pc.SetPoints(points);
    pc.SetNormals(normals);
    pc.SetColors(colors);

    ExpectEQ(Vector3f(19.607843, 0.0, 0.0), pc.GetMinBound());
    ExpectEQ(Vector3f(996.078431, 996.078431, 996.078431), pc.GetMaxBound());

    EXPECT_FALSE(pc.IsEmpty());
    EXPECT_TRUE(pc.HasPoints());
    EXPECT_TRUE(pc.HasNormals());
    EXPECT_TRUE(pc.HasColors());

    pc.Clear();

    // public members
    EXPECT_TRUE(pc.IsEmpty());

    ExpectEQ(Zero3f, pc.GetMinBound());
    ExpectEQ(Zero3f, pc.GetMaxBound());

    EXPECT_FALSE(pc.HasPoints());
    EXPECT_FALSE(pc.HasNormals());
    EXPECT_FALSE(pc.HasColors());
}

TEST(PointCloud, IsEmpty) {
    int size = 100;

    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(1000.0, 1000.0, 1000.0);

    geometry::PointCloud pc;

    EXPECT_TRUE(pc.IsEmpty());

    thrust::host_vector<Vector3f_u> points(size);
    Rand(points, vmin, vmax, 0);
    pc.SetPoints(points);

    EXPECT_FALSE(pc.IsEmpty());
}

TEST(PointCloud, GetMinBound) {
    int size = 100;

    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(1000.0, 1000.0, 1000.0);

    geometry::PointCloud pc;

    thrust::host_vector<Vector3f_u> points(size);
    Rand(points, vmin, vmax, 0);
    pc.SetPoints(points);

    ExpectEQ(Vector3f(19.607843, 0.0, 0.0), pc.GetMinBound());
    ExpectEQ(Vector3f(19.607843, 0.0, 0.0), pc.GetMinBound());
}

TEST(PointCloud, GetMaxBound) {
    int size = 100;

    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(1000.0, 1000.0, 1000.0);

    geometry::PointCloud pc;

    thrust::host_vector<Vector3f_u> points(size);
    Rand(points, vmin, vmax, 0);
    pc.SetPoints(points);

    ExpectEQ(Vector3f(996.078431, 996.078431, 996.078431), pc.GetMaxBound());
    ExpectEQ(Vector3f(996.078431, 996.078431, 996.078431), pc.GetMaxBound());
}

TEST(PointCloud, Transform) {
    thrust::host_vector<Vector3f_u> ref_points;
    ref_points.push_back(Vector3f_u(1.411252, 4.274168, 3.130918));
    ref_points.push_back(Vector3f_u(1.231757, 4.154505, 3.183678));
    ref_points.push_back(Vector3f_u(1.403168, 4.268779, 2.121679));
    ref_points.push_back(Vector3f_u(1.456767, 4.304511, 2.640845));
    ref_points.push_back(Vector3f_u(1.620902, 4.413935, 1.851255));
    ref_points.push_back(Vector3f_u(1.374684, 4.249790, 3.062485));
    ref_points.push_back(Vector3f_u(1.328160, 4.218773, 1.795728));
    ref_points.push_back(Vector3f_u(1.713446, 4.475631, 1.860145));
    ref_points.push_back(Vector3f_u(1.409239, 4.272826, 2.011462));
    ref_points.push_back(Vector3f_u(1.480169, 4.320113, 1.177780));

    thrust::host_vector<Vector3f_u> ref_normals;
    ref_normals.push_back(Vector3f_u(396.470588, 1201.176471, 880.352941));
    ref_normals.push_back(Vector3f_u(320.392157, 1081.176471, 829.019608));
    ref_normals.push_back(Vector3f_u(268.627451, 817.647059, 406.666667));
    ref_normals.push_back(Vector3f_u(338.431373, 1000.392157, 614.117647));
    ref_normals.push_back(Vector3f_u(423.137255, 1152.549020, 483.607843));
    ref_normals.push_back(Vector3f_u(432.549020, 1337.647059, 964.392157));
    ref_normals.push_back(Vector3f_u(139.607843, 443.921569, 189.176471));
    ref_normals.push_back(Vector3f_u(291.764706, 762.352941, 317.058824));
    ref_normals.push_back(Vector3f_u(134.117647, 407.058824, 191.882353));
    ref_normals.push_back(Vector3f_u(274.509804, 801.568627, 218.627451));

    int size = 10;
    geometry::PointCloud pc;

    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(1000.0, 1000.0, 1000.0);

    thrust::host_vector<Vector3f_u> points(size);
    Rand(points, vmin, vmax, 0);
    pc.SetPoints(points);

    thrust::host_vector<Vector3f_u> normals(size);
    normals.resize(size);
    Rand(normals, vmin, vmax, 0);
    pc.SetNormals(normals);

    Matrix4f transformation;
    transformation << 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16;

    pc.Transform(transformation);

    ExpectEQ(ref_points, pc.GetPoints());
    ExpectEQ(ref_normals, pc.GetNormals());
}

TEST(PointCloud, HasPoints) {
    int size = 100;

    geometry::PointCloud pc;

    EXPECT_FALSE(pc.HasPoints());

    thrust::host_vector<Vector3f_u> points(size);
    pc.SetPoints(points);

    EXPECT_TRUE(pc.HasPoints());
}

TEST(PointCloud, HasNormals) {
    int size = 100;

    geometry::PointCloud pc;

    EXPECT_FALSE(pc.HasNormals());

    thrust::host_vector<Vector3f_u> points(size);
    pc.SetPoints(points);
    thrust::host_vector<Vector3f_u> normals(size);
    pc.SetNormals(normals);

    EXPECT_TRUE(pc.HasNormals());
}

TEST(PointCloud, HasColors) {
    int size = 100;

    geometry::PointCloud pc;

    EXPECT_FALSE(pc.HasColors());

    thrust::host_vector<Vector3f_u> points(size);
    pc.SetPoints(points);
    thrust::host_vector<Vector3f_u> colors(size);
    pc.SetColors(colors);

    EXPECT_TRUE(pc.HasColors());
}

TEST(PointCloud, NormalizeNormals) {
    thrust::host_vector<Vector3f_u> ref;
    ref.push_back(Vector3f_u(0.692861, 0.323767, 0.644296));
    ref.push_back(Vector3f_u(0.650010, 0.742869, 0.160101));
    ref.push_back(Vector3f_u(0.379563, 0.870761, 0.312581));
    ref.push_back(Vector3f_u(0.575046, 0.493479, 0.652534));
    ref.push_back(Vector3f_u(0.320665, 0.448241, 0.834418));
    ref.push_back(Vector3f_u(0.691127, 0.480526, 0.539850));
    ref.push_back(Vector3f_u(0.227557, 0.973437, 0.025284));
    ref.push_back(Vector3f_u(0.281666, 0.156994, 0.946582));
    ref.push_back(Vector3f_u(0.341869, 0.894118, 0.289273));
    ref.push_back(Vector3f_u(0.103335, 0.972118, 0.210498));
    ref.push_back(Vector3f_u(0.441745, 0.723783, 0.530094));
    ref.push_back(Vector3f_u(0.336903, 0.727710, 0.597441));
    ref.push_back(Vector3f_u(0.434917, 0.862876, 0.257471));
    ref.push_back(Vector3f_u(0.636619, 0.435239, 0.636619));
    ref.push_back(Vector3f_u(0.393717, 0.876213, 0.277918));
    ref.push_back(Vector3f_u(0.275051, 0.633543, 0.723167));
    ref.push_back(Vector3f_u(0.061340, 0.873191, 0.483503));
    ref.push_back(Vector3f_u(0.118504, 0.276510, 0.953677));
    ref.push_back(Vector3f_u(0.930383, 0.360677, 0.065578));
    ref.push_back(Vector3f_u(0.042660, 0.989719, 0.136513));

    int size = 20;

    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(1000.0, 1000.0, 1000.0);

    geometry::PointCloud pc;

    thrust::host_vector<Vector3f_u> normals(size);
    Rand(normals, vmin, vmax, 0);
    pc.SetNormals(normals);

    pc.NormalizeNormals();

    ExpectEQ(ref, pc.GetNormals());
}

TEST(PointCloud, SelectDownSample) {
    thrust::host_vector<Vector3f_u> ref;
    ref.push_back(Vector3f_u(796.078431, 909.803922, 196.078431));
    ref.push_back(Vector3f_u(768.627451, 525.490196, 768.627451));
    ref.push_back(Vector3f_u(400.000000, 890.196078, 282.352941));
    ref.push_back(Vector3f_u(349.019608, 803.921569, 917.647059));
    ref.push_back(Vector3f_u(19.607843, 454.901961, 62.745098));
    ref.push_back(Vector3f_u(666.666667, 529.411765, 39.215686));
    ref.push_back(Vector3f_u(164.705882, 439.215686, 878.431373));
    ref.push_back(Vector3f_u(909.803922, 482.352941, 215.686275));
    ref.push_back(Vector3f_u(615.686275, 278.431373, 784.313725));
    ref.push_back(Vector3f_u(415.686275, 168.627451, 905.882353));
    ref.push_back(Vector3f_u(949.019608, 50.980392, 517.647059));
    ref.push_back(Vector3f_u(639.215686, 756.862745, 90.196078));
    ref.push_back(Vector3f_u(203.921569, 886.274510, 121.568627));
    ref.push_back(Vector3f_u(356.862745, 549.019608, 576.470588));
    ref.push_back(Vector3f_u(529.411765, 756.862745, 301.960784));
    ref.push_back(Vector3f_u(992.156863, 576.470588, 874.509804));
    ref.push_back(Vector3f_u(227.450980, 698.039216, 313.725490));
    ref.push_back(Vector3f_u(470.588235, 592.156863, 941.176471));
    ref.push_back(Vector3f_u(431.372549, 0.000000, 341.176471));
    ref.push_back(Vector3f_u(596.078431, 831.372549, 231.372549));
    ref.push_back(Vector3f_u(674.509804, 482.352941, 478.431373));
    ref.push_back(Vector3f_u(694.117647, 670.588235, 635.294118));
    ref.push_back(Vector3f_u(109.803922, 360.784314, 576.470588));
    ref.push_back(Vector3f_u(592.156863, 662.745098, 286.274510));
    ref.push_back(Vector3f_u(823.529412, 329.411765, 184.313725));

    size_t size = 100;
    geometry::PointCloud pc;

    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(1000.0, 1000.0, 1000.0);

    thrust::host_vector<Vector3f_u> points(size);
    Rand(points, vmin, vmax, 0);
    pc.SetPoints(points);

    thrust::host_vector<size_t> indices(size / 4);
    Rand(indices, 0, size, 0);

    // remove duplicates
    thrust::host_vector<size_t>::iterator it;
    it = thrust::unique(indices.begin(), indices.end());
    indices.resize(thrust::distance(indices.begin(), it));

    auto output_pc = pc.SelectDownSample(indices);
    auto output_pt = output_pc->GetPoints();

    sort::Do(ref);
    sort::Do(output_pt);
    ExpectEQ(ref, output_pt);
}

TEST(PointCloud, VoxelDownSample) {
    thrust::host_vector<Vector3f_u> ref_points;
    ref_points.push_back(Vector3f_u(19.607843, 454.901961, 62.745098));
    ref_points.push_back(Vector3f_u(66.666667, 949.019608, 525.490196));
    ref_points.push_back(Vector3f_u(82.352941, 192.156863, 662.745098));
    ref_points.push_back(Vector3f_u(105.882353, 996.078431, 215.686275));
    ref_points.push_back(Vector3f_u(141.176471, 603.921569, 15.686275));
    ref_points.push_back(Vector3f_u(152.941176, 400.000000, 129.411765));
    ref_points.push_back(Vector3f_u(239.215686, 133.333333, 803.921569));
    ref_points.push_back(Vector3f_u(294.117647, 635.294118, 521.568627));
    ref_points.push_back(Vector3f_u(333.333333, 764.705882, 274.509804));
    ref_points.push_back(Vector3f_u(349.019608, 803.921569, 917.647059));
    ref_points.push_back(Vector3f_u(364.705882, 509.803922, 949.019608));
    ref_points.push_back(Vector3f_u(400.000000, 890.196078, 282.352941));
    ref_points.push_back(Vector3f_u(490.196078, 972.549020, 290.196078));
    ref_points.push_back(Vector3f_u(509.803922, 835.294118, 611.764706));
    ref_points.push_back(Vector3f_u(552.941176, 474.509804, 627.450980));
    ref_points.push_back(Vector3f_u(768.627451, 525.490196, 768.627451));
    ref_points.push_back(Vector3f_u(796.078431, 909.803922, 196.078431));
    ref_points.push_back(Vector3f_u(839.215686, 392.156863, 780.392157));
    ref_points.push_back(Vector3f_u(890.196078, 345.098039, 62.745098));
    ref_points.push_back(Vector3f_u(913.725490, 635.294118, 713.725490));

    thrust::host_vector<Vector3f_u> ref_normals;
    ref_normals.push_back(Vector3f_u(0.042660, 0.989719, 0.136513));
    ref_normals.push_back(Vector3f_u(0.061340, 0.873191, 0.483503));
    ref_normals.push_back(Vector3f_u(0.103335, 0.972118, 0.210498));
    ref_normals.push_back(Vector3f_u(0.118504, 0.276510, 0.953677));
    ref_normals.push_back(Vector3f_u(0.227557, 0.973437, 0.025284));
    ref_normals.push_back(Vector3f_u(0.275051, 0.633543, 0.723167));
    ref_normals.push_back(Vector3f_u(0.281666, 0.156994, 0.946582));
    ref_normals.push_back(Vector3f_u(0.320665, 0.448241, 0.834418));
    ref_normals.push_back(Vector3f_u(0.336903, 0.727710, 0.597441));
    ref_normals.push_back(Vector3f_u(0.341869, 0.894118, 0.289273));
    ref_normals.push_back(Vector3f_u(0.379563, 0.870761, 0.312581));
    ref_normals.push_back(Vector3f_u(0.393717, 0.876213, 0.277918));
    ref_normals.push_back(Vector3f_u(0.434917, 0.862876, 0.257471));
    ref_normals.push_back(Vector3f_u(0.441745, 0.723783, 0.530094));
    ref_normals.push_back(Vector3f_u(0.575046, 0.493479, 0.652534));
    ref_normals.push_back(Vector3f_u(0.636619, 0.435239, 0.636619));
    ref_normals.push_back(Vector3f_u(0.650010, 0.742869, 0.160101));
    ref_normals.push_back(Vector3f_u(0.691127, 0.480526, 0.539850));
    ref_normals.push_back(Vector3f_u(0.692861, 0.323767, 0.644296));
    ref_normals.push_back(Vector3f_u(0.930383, 0.360677, 0.065578));

    thrust::host_vector<Vector3f_u> ref_colors;
    ref_colors.push_back(Vector3f_u(5.000000, 116.000000, 16.000000));
    ref_colors.push_back(Vector3f_u(17.000000, 242.000000, 134.000000));
    ref_colors.push_back(Vector3f_u(21.000000, 49.000000, 169.000000));
    ref_colors.push_back(Vector3f_u(27.000000, 254.000000, 55.000000));
    ref_colors.push_back(Vector3f_u(36.000000, 154.000000, 4.000000));
    ref_colors.push_back(Vector3f_u(39.000000, 102.000000, 33.000000));
    ref_colors.push_back(Vector3f_u(61.000000, 34.000000, 205.000000));
    ref_colors.push_back(Vector3f_u(75.000000, 162.000000, 133.000000));
    ref_colors.push_back(Vector3f_u(85.000000, 195.000000, 70.000000));
    ref_colors.push_back(Vector3f_u(89.000000, 205.000000, 234.000000));
    ref_colors.push_back(Vector3f_u(93.000000, 130.000000, 242.000000));
    ref_colors.push_back(Vector3f_u(102.000000, 227.000000, 72.000000));
    ref_colors.push_back(Vector3f_u(125.000000, 248.000000, 74.000000));
    ref_colors.push_back(Vector3f_u(130.000000, 213.000000, 156.000000));
    ref_colors.push_back(Vector3f_u(141.000000, 121.000000, 160.000000));
    ref_colors.push_back(Vector3f_u(196.000000, 134.000000, 196.000000));
    ref_colors.push_back(Vector3f_u(203.000000, 232.000000, 50.000000));
    ref_colors.push_back(Vector3f_u(214.000000, 100.000000, 199.000000));
    ref_colors.push_back(Vector3f_u(227.000000, 88.000000, 16.000000));
    ref_colors.push_back(Vector3f_u(233.000000, 162.000000, 182.000000));

    size_t size = 20;
    geometry::PointCloud pc;

    thrust::host_vector<Vector3f_u> points(size);
    Rand(points, Zero3f, Vector3f(1000.0, 1000.0, 1000.0), 0);
    pc.SetPoints(points);
    thrust::host_vector<Vector3f_u> normals(size);
    Rand(normals, Zero3f, Vector3f(10.0, 10.0, 10.0), 0);
    pc.SetNormals(normals);
    thrust::host_vector<Vector3f_u> colors(size);
    Rand(colors, Zero3f, Vector3f(255.0, 255.0, 255.0), 0);
    pc.SetColors(colors);

    float voxel_size = 0.5;
    auto output_pc = pc.VoxelDownSample(voxel_size);

    // sometimes the order of these Vector3d values can be mixed-up
    // sort these vectors in order to match the expected order.
    auto output_pt = output_pc->GetPoints();
    auto output_nl = output_pc->GetNormals();
    auto output_cl = output_pc->GetColors();
    sort::Do(ref_points);
    sort::Do(ref_normals);
    sort::Do(ref_colors);
    sort::Do(output_pt);
    sort::Do(output_nl);
    sort::Do(output_cl);

    ExpectEQ(ref_points, output_pt);
    ExpectEQ(ref_normals, output_nl);
    ExpectEQ(ref_colors, output_cl);
}