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
#include "cupoch/geometry/trianglemesh.h"

#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/pointcloud.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

void ExpectEQ(const geometry::TriangleMesh& mesh0,
              const geometry::TriangleMesh& mesh1,
              float threshold = 1e-4) {
    ExpectEQ(mesh0.GetVertices(), mesh1.GetVertices(), threshold);
    ExpectEQ(mesh0.GetVertexNormals(), mesh1.GetVertexNormals(), threshold);
    ExpectEQ(mesh0.GetVertexColors(), mesh1.GetVertexColors(), threshold);
    ExpectEQ(mesh0.GetTriangles(), mesh1.GetTriangles());
    ExpectEQ(mesh0.GetTriangleNormals(), mesh1.GetTriangleNormals(), threshold);
}

TEST(TriangleMesh, Constructor) {
    geometry::TriangleMesh tm;

    // inherited from Geometry2D
    EXPECT_EQ(geometry::Geometry::GeometryType::TriangleMesh,
              tm.GetGeometryType());
    EXPECT_EQ(3, tm.Dimension());

    // public member variables
    EXPECT_EQ(0u, tm.vertices_.size());
    EXPECT_EQ(0u, tm.vertex_normals_.size());
    EXPECT_EQ(0u, tm.vertex_colors_.size());
    EXPECT_EQ(0u, tm.triangles_.size());
    EXPECT_EQ(0u, tm.triangle_normals_.size());

    // public members
    EXPECT_TRUE(tm.IsEmpty());

    ExpectEQ(Zero3f, tm.GetMinBound());
    ExpectEQ(Zero3f, tm.GetMaxBound());

    EXPECT_FALSE(tm.HasVertices());
    EXPECT_FALSE(tm.HasVertexNormals());
    EXPECT_FALSE(tm.HasVertexColors());
    EXPECT_FALSE(tm.HasTriangles());
    EXPECT_FALSE(tm.HasTriangleNormals());
}

TEST(TriangleMesh, DISABLED_MemberData) { unit_test::NotImplemented(); }

TEST(TriangleMesh, Clear) {
    int size = 100;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(1000.0, 1000.0, 1000.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    std::vector<Vector3f> vertices(size);
    std::vector<Vector3f> vertex_normals(size);
    std::vector<Vector3f> vertex_colors(size);
    std::vector<Vector3i> triangles(size);
    std::vector<Vector3f> triangle_normals(size);

    Rand(vertices, dmin, dmax, 0);
    tm.SetVertices(vertices);
    Rand(vertex_normals, dmin, dmax, 0);
    tm.SetVertexNormals(vertex_normals);
    Rand(vertex_colors, dmin, dmax, 0);
    tm.SetVertexColors(vertex_colors);
    Rand(triangles, imin, imax, 0);
    tm.SetTriangles(triangles);
    Rand(triangle_normals, dmin, dmax, 0);
    tm.SetTriangleNormals(triangle_normals);

    EXPECT_FALSE(tm.IsEmpty());

    ExpectEQ(Vector3f(19.607843, 0.0, 0.0), tm.GetMinBound());
    ExpectEQ(Vector3f(996.078431, 996.078431, 996.078431), tm.GetMaxBound());

    EXPECT_TRUE(tm.HasVertices());
    EXPECT_TRUE(tm.HasVertexNormals());
    EXPECT_TRUE(tm.HasVertexColors());
    EXPECT_TRUE(tm.HasTriangles());
    EXPECT_TRUE(tm.HasTriangleNormals());

    tm.Clear();

    // public members
    EXPECT_TRUE(tm.IsEmpty());

    ExpectEQ(Zero3f, tm.GetMinBound());
    ExpectEQ(Zero3f, tm.GetMaxBound());

    EXPECT_FALSE(tm.HasVertices());
    EXPECT_FALSE(tm.HasVertexNormals());
    EXPECT_FALSE(tm.HasVertexColors());
    EXPECT_FALSE(tm.HasTriangles());
    EXPECT_FALSE(tm.HasTriangleNormals());
}

TEST(TriangleMesh, IsEmpty) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_TRUE(tm.IsEmpty());

    std::vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);

    EXPECT_FALSE(tm.IsEmpty());
}

TEST(TriangleMesh, GetMinBound) {
    int size = 100;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(1000.0, 1000.0, 1000.0);

    geometry::TriangleMesh tm;

    std::vector<Vector3f> vertices(size);
    Rand(vertices, dmin, dmax, 0);
    tm.SetVertices(vertices);

    ExpectEQ(Vector3f(19.607843, 0.0, 0.0), tm.GetMinBound());
}

TEST(TriangleMesh, GetMaxBound) {
    int size = 100;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(1000.0, 1000.0, 1000.0);

    geometry::TriangleMesh tm;

    std::vector<Vector3f> vertices(size);
    Rand(vertices, dmin, dmax, 0);
    tm.SetVertices(vertices);

    ExpectEQ(Vector3f(996.078431, 996.078431, 996.078431), tm.GetMaxBound());
}

TEST(TriangleMesh, OperatorAppend) {
    size_t size = 100;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(1000.0, 1000.0, 1000.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm0;
    geometry::TriangleMesh tm1;

    std::vector<Vector3f> vertices0(size);
    std::vector<Vector3f> vertex_normals0(size);
    std::vector<Vector3f> vertex_colors0(size);
    std::vector<Vector3i> triangles0(size);
    std::vector<Vector3f> triangle_normals0(size);
    Rand(vertices0, dmin, dmax, 0);
    Rand(vertex_normals0, dmin, dmax, 0);
    Rand(vertex_colors0, dmin, dmax, 0);
    Rand(triangles0, imin, imax, 0);
    Rand(triangle_normals0, dmin, dmax, 0);
    tm0.SetVertices(vertices0);
    tm0.SetVertexNormals(vertex_normals0);
    tm0.SetVertexColors(vertex_colors0);
    tm0.SetTriangles(triangles0);
    tm0.SetTriangleNormals(triangle_normals0);

    std::vector<Vector3f> vertices1(size);
    std::vector<Vector3f> vertex_normals1(size);
    std::vector<Vector3f> vertex_colors1(size);
    std::vector<Vector3i> triangles1(size);
    std::vector<Vector3f> triangle_normals1(size);
    Rand(vertices1, dmin, dmax, 0);
    Rand(vertex_normals1, dmin, dmax, 0);
    Rand(vertex_colors1, dmin, dmax, 0);
    Rand(triangles1, imin, imax, 0);
    Rand(triangle_normals1, dmin, dmax, 0);
    tm1.SetVertices(vertices1);
    tm1.SetVertexNormals(vertex_normals1);
    tm1.SetVertexColors(vertex_colors1);
    tm1.SetTriangles(triangles1);
    tm1.SetTriangleNormals(triangle_normals1);

    geometry::TriangleMesh tm(tm0);
    tm += tm1;

    std::vector<Vector3f> vertices = tm.GetVertices();
    std::vector<Vector3f> vertex_normals = tm.GetVertexNormals();
    std::vector<Vector3f> vertex_colors = tm.GetVertexColors();
    std::vector<Vector3i> triangles = tm.GetTriangles();
    std::vector<Vector3f> triangle_normals = tm.GetTriangleNormals();

    EXPECT_EQ(2 * size, tm.vertices_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(vertices0[i], vertices[i + 0]);
        ExpectEQ(vertices1[i], vertices[i + size]);
    }

    EXPECT_EQ(2 * size, tm.vertex_normals_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(vertex_normals0[i], vertex_normals[i + 0]);
        ExpectEQ(vertex_normals1[i], vertex_normals[i + size]);
    }

    EXPECT_EQ(2 * size, tm.vertex_colors_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(vertex_colors0[i], vertex_colors[i + 0]);
        ExpectEQ(vertex_colors1[i], vertex_colors[i + size]);
    }

    // NOTE: why is this offset required only for triangles?
    EXPECT_EQ(2 * size, tm.triangles_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(triangles0[i], triangles[i + 0]);
        ExpectEQ(
                Vector3i(triangles1[i](0, 0) + size, triangles1[i](1, 0) + size,
                         triangles1[i](2, 0) + size),
                triangles[i + size]);
    }

    EXPECT_EQ(2 * size, tm.triangle_normals_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(triangle_normals0[i], triangle_normals[i + 0]);
        ExpectEQ(triangle_normals1[i], triangle_normals[i + size]);
    }
}

TEST(TriangleMesh, OperatorADD) {
    size_t size = 100;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(1000.0, 1000.0, 1000.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm0;
    geometry::TriangleMesh tm1;

    std::vector<Vector3f> vertices0(size);
    std::vector<Vector3f> vertex_normals0(size);
    std::vector<Vector3f> vertex_colors0(size);
    std::vector<Vector3i> triangles0(size);
    std::vector<Vector3f> triangle_normals0(size);
    Rand(vertices0, dmin, dmax, 0);
    Rand(vertex_normals0, dmin, dmax, 0);
    Rand(vertex_colors0, dmin, dmax, 0);
    Rand(triangles0, imin, imax, 0);
    Rand(triangle_normals0, dmin, dmax, 0);
    tm0.SetVertices(vertices0);
    tm0.SetVertexNormals(vertex_normals0);
    tm0.SetVertexColors(vertex_colors0);
    tm0.SetTriangles(triangles0);
    tm0.SetTriangleNormals(triangle_normals0);

    std::vector<Vector3f> vertices1(size);
    std::vector<Vector3f> vertex_normals1(size);
    std::vector<Vector3f> vertex_colors1(size);
    std::vector<Vector3i> triangles1(size);
    std::vector<Vector3f> triangle_normals1(size);
    Rand(vertices1, dmin, dmax, 0);
    Rand(vertex_normals1, dmin, dmax, 0);
    Rand(vertex_colors1, dmin, dmax, 0);
    Rand(triangles1, imin, imax, 0);
    Rand(triangle_normals1, dmin, dmax, 0);
    tm1.SetVertices(vertices1);
    tm1.SetVertexNormals(vertex_normals1);
    tm1.SetVertexColors(vertex_colors1);
    tm1.SetTriangles(triangles1);
    tm1.SetTriangleNormals(triangle_normals1);

    geometry::TriangleMesh tm = tm0 + tm1;

    std::vector<Vector3f> vertices = tm.GetVertices();
    std::vector<Vector3f> vertex_normals = tm.GetVertexNormals();
    std::vector<Vector3f> vertex_colors = tm.GetVertexColors();
    std::vector<Vector3i> triangles = tm.GetTriangles();
    std::vector<Vector3f> triangle_normals = tm.GetTriangleNormals();

    EXPECT_EQ(2 * size, tm.vertices_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(vertices0[i], vertices[i + 0]);
        ExpectEQ(vertices1[i], vertices[i + size]);
    }

    EXPECT_EQ(2 * size, tm.vertex_normals_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(vertex_normals0[i], vertex_normals[i + 0]);
        ExpectEQ(vertex_normals1[i], vertex_normals[i + size]);
    }

    EXPECT_EQ(2 * size, tm.vertex_colors_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(vertex_colors0[i], vertex_colors[i + 0]);
        ExpectEQ(vertex_colors1[i], vertex_colors[i + size]);
    }

    // NOTE: why is this offset required only for triangles?
    EXPECT_EQ(2 * size, tm.triangles_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(triangles0[i], triangles[i + 0]);
        ExpectEQ(
                Vector3i(triangles1[i](0, 0) + size, triangles1[i](1, 0) + size,
                         triangles1[i](2, 0) + size),
                triangles[i + size]);
    }

    EXPECT_EQ(2 * size, tm.triangle_normals_.size());
    for (size_t i = 0; i < size; i++) {
        ExpectEQ(triangle_normals0[i], triangle_normals[i + 0]);
        ExpectEQ(triangle_normals1[i], triangle_normals[i + size]);
    }
}

TEST(TriangleMesh, ComputeTriangleNormals) {
    Vector3f ref_raw[] = {{-0.119231, 0.738792, 0.663303},
                          {-0.115181, 0.730934, 0.672658},
                          {-0.589738, -0.764139, -0.261344},
                          {-0.330250, 0.897644, -0.291839},
                          {-0.164192, 0.976753, 0.137819},
                          {-0.475702, 0.727947, 0.493762},
                          {0.990884, -0.017339, -0.133596},
                          {0.991673, 0.091418, -0.090700},
                          {0.722410, 0.154580, -0.673965},
                          {0.598552, -0.312929, -0.737435},
                          {0.712875, -0.628251, -0.311624},
                          {0.233815, -0.638800, -0.732984},
                          {0.494773, -0.472428, -0.729391},
                          {0.583861, 0.796905, 0.155075},
                          {-0.277650, -0.948722, -0.151119},
                          {-0.791337, 0.093176, 0.604238},
                          {0.569287, 0.158108, 0.806793},
                          {0.115315, 0.914284, 0.388314},
                          {0.105421, 0.835841, -0.538754},
                          {0.473326, 0.691900, -0.545195},
                          {0.719515, 0.684421, -0.117757},
                          {-0.713642, -0.691534, -0.111785},
                          {-0.085377, -0.916925, 0.389820},
                          {0.787892, 0.611808, -0.070127},
                          {0.788022, 0.488544, 0.374628}};
    std::vector<Vector3f> ref(25);
    for (int i = 0; i < 25; ++i) ref[i] = ref_raw[i];

    size_t size = 25;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(10.0, 10.0, 10.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    std::vector<Vector3f> vertices(size);
    Rand(vertices, dmin, dmax, 0);
    tm.SetVertices(vertices);

    std::vector<Vector3i> triangles;
    for (size_t i = 0; i < size; i++)
        triangles.push_back(Vector3i(i, (i + 1) % size, (i + 2) % size));
    tm.SetTriangles(triangles);

    tm.ComputeTriangleNormals();

    ExpectEQ(ref, tm.GetTriangleNormals());
}

TEST(TriangleMesh, ComputeVertexNormals) {
    Vector3f ref_raw[] = {
            {0.635868, 0.698804, 0.327636},    {0.327685, 0.717012, 0.615237},
            {-0.346072, 0.615418, 0.708163},   {-0.690485, 0.663544, 0.287992},
            {-0.406664, 0.913428, -0.016549},  {-0.356568, 0.888296, 0.289466},
            {-0.276491, 0.894931, 0.350216},   {0.262855, 0.848183, 0.459883},
            {0.933461, 0.108347, -0.341923},   {0.891804, 0.050667, -0.449577},
            {0.735392, -0.110383, -0.668592},  {0.469090, -0.564602, -0.679102},
            {0.418223, -0.628548, -0.655758},  {0.819226, 0.168537, -0.548145},
            {0.963613, 0.103044, -0.246642},   {-0.506244, 0.320837, 0.800488},
            {0.122226, -0.058031, 0.990804},   {0.175502, 0.533543, 0.827364},
            {0.384132, 0.892338, 0.237015},    {0.273664, 0.896739, -0.347804},
            {0.361530, 0.784805, -0.503366},   {0.429700, 0.646636, -0.630253},
            {-0.264834, -0.963970, -0.025005}, {0.940214, -0.336158, -0.054732},
            {0.862650, 0.449603, 0.231714}};
    std::vector<Vector3f> ref(25);
    for (int i = 0; i < 25; ++i) ref[i] = ref_raw[i];

    size_t size = 25;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(10.0, 10.0, 10.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    std::vector<Vector3f> vertices(size);
    Rand(vertices, dmin, dmax, 0);
    tm.SetVertices(vertices);

    std::vector<Vector3i> triangles;
    for (size_t i = 0; i < size; i++)
        triangles.push_back(Vector3i(i, (i + 1) % size, (i + 2) % size));
    tm.SetTriangles(triangles);

    tm.ComputeVertexNormals();

    ExpectEQ(ref, tm.GetVertexNormals());
}

TEST(TriangleMesh, ComputeEdgeList) {
    // 4-sided pyramid with A as top vertex, bottom has two triangles
    Eigen::Vector3f A(0, 0, 1);    // 0
    Eigen::Vector3f B(1, 1, 0);    // 1
    Eigen::Vector3f C(-1, 1, 0);   // 2
    Eigen::Vector3f D(-1, -1, 0);  // 3
    Eigen::Vector3f E(1, -1, 0);   // 4
    std::vector<Eigen::Vector3f> vertices;
    vertices.push_back(A);
    vertices.push_back(B);
    vertices.push_back(C);
    vertices.push_back(D);
    vertices.push_back(E);

    geometry::TriangleMesh tm;
    tm.SetVertices(vertices);
    std::vector<Eigen::Vector3i> triangles;
    triangles.push_back(Eigen::Vector3i(0, 1, 2));
    triangles.push_back(Eigen::Vector3i(0, 2, 3));
    triangles.push_back(Eigen::Vector3i(0, 3, 4));
    triangles.push_back(Eigen::Vector3i(0, 4, 1));
    triangles.push_back(Eigen::Vector3i(1, 2, 4));
    triangles.push_back(Eigen::Vector3i(2, 3, 4));
    tm.SetTriangles(triangles);
    EXPECT_FALSE(tm.HasEdgeList());
    tm.ComputeEdgeList();
    EXPECT_TRUE(tm.HasEdgeList());

    std::vector<Eigen::Vector2i> ref;
    ref.push_back({0, 1});
    ref.push_back({0, 2});
    ref.push_back({0, 3});
    ref.push_back({0, 4});
    ref.push_back({1, 0});
    ref.push_back({1, 2});
    ref.push_back({1, 4});
    ref.push_back({2, 0});
    ref.push_back({2, 1});
    ref.push_back({2, 3});
    ref.push_back({2, 4});
    ref.push_back({3, 0});
    ref.push_back({3, 2});
    ref.push_back({3, 4});
    ref.push_back({4, 0});
    ref.push_back({4, 1});
    ref.push_back({4, 2});
    ref.push_back({4, 3});
    ExpectEQ(ref, tm.GetEdgeList());
}

TEST(TriangleMesh, Purge) {
    Vector3f ref_vertices_raw[] = {{839.215686, 392.156863, 780.392157},
                                   {796.078431, 909.803922, 196.078431},
                                   {333.333333, 764.705882, 274.509804},
                                   {552.941176, 474.509804, 627.450980},
                                   {364.705882, 509.803922, 949.019608},
                                   {913.725490, 635.294118, 713.725490},
                                   {141.176471, 603.921569, 15.686275},
                                   {239.215686, 133.333333, 803.921569},
                                   {152.941176, 400.000000, 129.411765},
                                   {105.882353, 996.078431, 215.686275},
                                   {509.803922, 835.294118, 611.764706},
                                   {294.117647, 635.294118, 521.568627},
                                   {490.196078, 972.549020, 290.196078},
                                   {768.627451, 525.490196, 768.627451},
                                   {400.000000, 890.196078, 282.352941},
                                   {349.019608, 803.921569, 917.647059},
                                   {66.666667, 949.019608, 525.490196},
                                   {82.352941, 192.156863, 662.745098},
                                   {890.196078, 345.098039, 62.745098},
                                   {19.607843, 454.901961, 62.745098},
                                   {235.294118, 968.627451, 901.960784},
                                   {847.058824, 262.745098, 537.254902},
                                   {372.549020, 756.862745, 509.803922},
                                   {666.666667, 529.411765, 39.215686}};
    std::vector<Vector3f> ref_vertices(24);
    for (int i = 0; i < 24; ++i) ref_vertices[i] = ref_vertices_raw[i];

    Vector3f ref_vertex_normals_raw[] = {{839.215686, 392.156863, 780.392157},
                                         {796.078431, 909.803922, 196.078431},
                                         {333.333333, 764.705882, 274.509804},
                                         {552.941176, 474.509804, 627.450980},
                                         {364.705882, 509.803922, 949.019608},
                                         {913.725490, 635.294118, 713.725490},
                                         {141.176471, 603.921569, 15.686275},
                                         {239.215686, 133.333333, 803.921569},
                                         {152.941176, 400.000000, 129.411765},
                                         {105.882353, 996.078431, 215.686275},
                                         {509.803922, 835.294118, 611.764706},
                                         {294.117647, 635.294118, 521.568627},
                                         {490.196078, 972.549020, 290.196078},
                                         {768.627451, 525.490196, 768.627451},
                                         {400.000000, 890.196078, 282.352941},
                                         {349.019608, 803.921569, 917.647059},
                                         {66.666667, 949.019608, 525.490196},
                                         {82.352941, 192.156863, 662.745098},
                                         {890.196078, 345.098039, 62.745098},
                                         {19.607843, 454.901961, 62.745098},
                                         {235.294118, 968.627451, 901.960784},
                                         {847.058824, 262.745098, 537.254902},
                                         {372.549020, 756.862745, 509.803922},
                                         {666.666667, 529.411765, 39.215686}};
    std::vector<Vector3f> ref_vertex_normals(24);
    for (int i = 0; i < 24; ++i)
        ref_vertex_normals[i] = ref_vertex_normals_raw[i];

    Vector3f ref_vertex_colors_raw[] = {{839.215686, 392.156863, 780.392157},
                                        {796.078431, 909.803922, 196.078431},
                                        {333.333333, 764.705882, 274.509804},
                                        {552.941176, 474.509804, 627.450980},
                                        {364.705882, 509.803922, 949.019608},
                                        {913.725490, 635.294118, 713.725490},
                                        {141.176471, 603.921569, 15.686275},
                                        {239.215686, 133.333333, 803.921569},
                                        {152.941176, 400.000000, 129.411765},
                                        {105.882353, 996.078431, 215.686275},
                                        {509.803922, 835.294118, 611.764706},
                                        {294.117647, 635.294118, 521.568627},
                                        {490.196078, 972.549020, 290.196078},
                                        {768.627451, 525.490196, 768.627451},
                                        {400.000000, 890.196078, 282.352941},
                                        {349.019608, 803.921569, 917.647059},
                                        {66.666667, 949.019608, 525.490196},
                                        {82.352941, 192.156863, 662.745098},
                                        {890.196078, 345.098039, 62.745098},
                                        {19.607843, 454.901961, 62.745098},
                                        {235.294118, 968.627451, 901.960784},
                                        {847.058824, 262.745098, 537.254902},
                                        {372.549020, 756.862745, 509.803922},
                                        {666.666667, 529.411765, 39.215686}};
    std::vector<Vector3f> ref_vertex_colors(24);
    for (int i = 0; i < 24; ++i)
        ref_vertex_colors[i] = ref_vertex_colors_raw[i];

    Vector3i ref_triangles_raw[] = {
            {20, 9, 18},  {19, 21, 4}, {8, 18, 6}, {13, 11, 15}, {8, 12, 22},
            {21, 15, 17}, {3, 14, 0},  {5, 3, 19}, {2, 23, 5},   {12, 20, 14},
            {7, 15, 12},  {11, 23, 6}, {9, 21, 6}, {8, 19, 22},  {1, 22, 12},
            {1, 4, 15},   {21, 8, 1},  {0, 10, 1}, {5, 23, 21},  {20, 6, 12},
            {8, 18, 12},  {16, 12, 0}};
    std::vector<Vector3i> ref_triangles(22);
    for (int i = 0; i < 22; ++i) ref_triangles[i] = ref_triangles_raw[i];

    Vector3f ref_triangle_normals_raw[] = {{839.215686, 392.156863, 780.392157},
                                           {796.078431, 909.803922, 196.078431},
                                           {333.333333, 764.705882, 274.509804},
                                           {552.941176, 474.509804, 627.450980},
                                           {364.705882, 509.803922, 949.019608},
                                           {913.725490, 635.294118, 713.725490},
                                           {141.176471, 603.921569, 15.686275},
                                           {239.215686, 133.333333, 803.921569},
                                           {105.882353, 996.078431, 215.686275},
                                           {509.803922, 835.294118, 611.764706},
                                           {294.117647, 635.294118, 521.568627},
                                           {490.196078, 972.549020, 290.196078},
                                           {400.000000, 890.196078, 282.352941},
                                           {349.019608, 803.921569, 917.647059},
                                           {66.666667, 949.019608, 525.490196},
                                           {82.352941, 192.156863, 662.745098},
                                           {890.196078, 345.098039, 62.745098},
                                           {19.607843, 454.901961, 62.745098},
                                           {235.294118, 968.627451, 901.960784},
                                           {847.058824, 262.745098, 537.254902},
                                           {372.549020, 756.862745, 509.803922},
                                           {666.666667, 529.411765, 39.215686}};
    std::vector<Vector3f> ref_triangle_normals(22);
    for (int i = 0; i < 22; ++i)
        ref_triangle_normals[i] = ref_triangle_normals_raw[i];

    int size = 25;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(1000.0, 1000.0, 1000.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm0;
    geometry::TriangleMesh tm1;

    std::vector<Vector3f> vertices(size);
    std::vector<Vector3f> vertex_normals(size);
    std::vector<Vector3f> vertex_colors(size);
    std::vector<Vector3i> triangles(size);
    std::vector<Vector3f> triangle_normals(size);
    Rand(vertices, dmin, dmax, 0);
    Rand(vertex_normals, dmin, dmax, 0);
    Rand(vertex_colors, dmin, dmax, 0);
    Rand(triangles, imin, imax, 0);
    Rand(triangle_normals, dmin, dmax, 0);

    tm0.SetVertices(vertices);
    tm0.SetVertexNormals(vertex_normals);
    tm0.SetVertexColors(vertex_colors);
    tm0.SetTriangles(triangles);
    tm0.SetTriangleNormals(triangle_normals);

    Rand(vertices, dmin, dmax, 0);
    Rand(vertex_normals, dmin, dmax, 0);
    Rand(vertex_colors, dmin, dmax, 1);
    Rand(triangles, imin, imax, 0);
    Rand(triangle_normals, dmin, dmax, 0);

    tm1.SetVertices(vertices);
    tm1.SetVertexNormals(vertex_normals);
    tm1.SetVertexColors(vertex_colors);
    tm1.SetTriangles(triangles);
    tm1.SetTriangleNormals(triangle_normals);

    geometry::TriangleMesh tm = tm0 + tm1;

    tm.RemoveDuplicatedVertices();
    tm.RemoveDuplicatedTriangles();
    tm.RemoveUnreferencedVertices();
    tm.RemoveDegenerateTriangles();

    {
        auto out = tm.GetVertices();
        sort::Do(ref_vertices);
        sort::Do(out);
        ExpectEQ(ref_vertices, out);
    }
    {
        auto out = tm.GetVertexNormals();
        sort::Do(ref_vertex_normals);
        sort::Do(out);
        ExpectEQ(ref_vertex_normals, out);
    }
    {
        auto out = tm.GetVertexColors();
        sort::Do(ref_vertex_colors);
        sort::Do(out);
        ExpectEQ(ref_vertex_colors, out);
    }

    EXPECT_EQ(ref_triangles.size(), tm.triangles_.size());

    {
        auto out = tm.GetTriangleNormals();
        sort::Do(ref_triangle_normals);
        sort::Do(out);
        ExpectEQ(ref_triangle_normals, out);
    }
}

TEST(TriangleMesh, SamplePointsUniformly) {
    auto mesh_empty = geometry::TriangleMesh();
    EXPECT_THROW(mesh_empty.SamplePointsUniformly(100), std::runtime_error);

    std::vector<Vector3f> vertices;
    vertices.push_back(Vector3f(0, 0, 0));
    vertices.push_back(Vector3f(1, 0, 0));
    vertices.push_back(Vector3f(0, 1, 0));
    std::vector<Vector3i> triangles;
    triangles.push_back(Vector3i(0, 1, 2));

    auto mesh_simple = geometry::TriangleMesh();
    mesh_simple.SetVertices(vertices);
    mesh_simple.SetTriangles(triangles);

    size_t n_points = 100;
    auto pcd_simple = mesh_simple.SamplePointsUniformly(n_points);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == 0);
    EXPECT_TRUE(pcd_simple->normals_.size() == 0);

    std::vector<Vector3f> colors;
    colors.push_back(Vector3f(1, 0, 0));
    colors.push_back(Vector3f(1, 0, 0));
    colors.push_back(Vector3f(1, 0, 0));
    std::vector<Vector3f> normals;
    normals.push_back(Vector3f(0, 1, 0));
    normals.push_back(Vector3f(0, 1, 0));
    normals.push_back(Vector3f(0, 1, 0));
    mesh_simple.SetVertexColors(colors);
    mesh_simple.SetVertexNormals(normals);
    pcd_simple = mesh_simple.SamplePointsUniformly(n_points);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == n_points);
    EXPECT_TRUE(pcd_simple->normals_.size() == n_points);

    std::vector<Vector3f> hc = pcd_simple->GetColors();
    std::vector<Vector3f> hn = pcd_simple->GetNormals();
    for (size_t pidx = 0; pidx < n_points; ++pidx) {
        ExpectEQ(hc[pidx], Vector3f(1, 0, 0));
        ExpectEQ(hn[pidx], Vector3f(0, 1, 0));
    }

    // use triangle normal instead of the vertex normals
    EXPECT_TRUE(mesh_simple.HasTriangleNormals() == false);
    pcd_simple = mesh_simple.SamplePointsUniformly(n_points, true);
    // the mesh now has triangle normals as a side effect.
    EXPECT_TRUE(mesh_simple.HasTriangleNormals() == true);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == n_points);
    EXPECT_TRUE(pcd_simple->normals_.size() == n_points);

    hc = pcd_simple->GetColors();
    hn = pcd_simple->GetNormals();
    for (size_t pidx = 0; pidx < n_points; ++pidx) {
        ExpectEQ(hc[pidx], Vector3f(1, 0, 0));
        ExpectEQ(hn[pidx], Vector3f(0, 0, 1));
    }

    // use triangle normal, this time the mesh has no vertex normals
    mesh_simple.SetVertexNormals(std::vector<Vector3f>());
    pcd_simple = mesh_simple.SamplePointsUniformly(n_points, true);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == n_points);
    EXPECT_TRUE(pcd_simple->normals_.size() == n_points);

    hc = pcd_simple->GetColors();
    hn = pcd_simple->GetNormals();
    for (size_t pidx = 0; pidx < n_points; ++pidx) {
        ExpectEQ(hc[pidx], Vector3f(1, 0, 0));
        ExpectEQ(hn[pidx], Vector3f(0, 0, 1));
    }
}

TEST(TriangleMesh, FilterSharpen) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3i> triangles;
    vertices.push_back({0, 0, 0});
    vertices.push_back({1, 0, 0});
    vertices.push_back({0, 1, 0});
    vertices.push_back({-1, 0, 0});
    vertices.push_back({0, -1, 0});
    triangles.push_back({0, 1, 2});
    triangles.push_back({0, 2, 3});
    triangles.push_back({0, 3, 4});
    triangles.push_back({0, 4, 1});
    mesh->SetVertices(vertices);
    mesh->SetTriangles(triangles);

    mesh = mesh->FilterSharpen(1, 1);
    std::vector<Eigen::Vector3f> ref1;
    ref1.push_back({0, 0, 0});
    ref1.push_back({4, 0, 0});
    ref1.push_back({0, 4, 0});
    ref1.push_back({-4, 0, 0});
    ref1.push_back({0, -4, 0});
    ExpectEQ(mesh->GetVertices(), ref1);

    mesh = mesh->FilterSharpen(9, 0.1);
    std::vector<Eigen::Vector3f> ref2;
    ref2.push_back({0, 0, 0});
    ref2.push_back({42.417997, 0, 0});
    ref2.push_back({0, 42.417997, 0});
    ref2.push_back({-42.417997, 0, 0});
    ref2.push_back({0, -42.417997, 0});
    ExpectEQ(mesh->GetVertices(), ref2);
}

TEST(TriangleMesh, FilterSmoothSimple) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3i> triangles;
    vertices.push_back({0, 0, 0});
    vertices.push_back({1, 0, 0});
    vertices.push_back({0, 1, 0});
    vertices.push_back({-1, 0, 0});
    vertices.push_back({0, -1, 0});
    triangles.push_back({0, 1, 2});
    triangles.push_back({0, 2, 3});
    triangles.push_back({0, 3, 4});
    triangles.push_back({0, 4, 1});
    mesh->SetVertices(vertices);
    mesh->SetTriangles(triangles);

    mesh = mesh->FilterSmoothSimple(1);
    std::vector<Eigen::Vector3f> ref1;
    ref1.push_back({0, 0, 0});
    ref1.push_back({0.25, 0, 0});
    ref1.push_back({0, 0.25, 0});
    ref1.push_back({-0.25, 0, 0});
    ref1.push_back({0, -0.25, 0});
    ExpectEQ(mesh->GetVertices(), ref1);

    mesh = mesh->FilterSmoothSimple(3);
    std::vector<Eigen::Vector3f> ref2;
    ref2.push_back({0, 0, 0});
    ref2.push_back({0.003906, 0, 0});
    ref2.push_back({0, 0.003906, 0});
    ref2.push_back({-0.003906, 0, 0});
    ref2.push_back({0, -0.003906, 0});
    ExpectEQ(mesh->GetVertices(), ref2);
}

TEST(TriangleMesh, FilterSmoothLaplacian) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    std::vector<Eigen::Vector3f> vertices;
    vertices.push_back({0, 0, 0});
    vertices.push_back({1, 0, 0});
    vertices.push_back({0, 1, 0});
    vertices.push_back({-1, 0, 0});
    vertices.push_back({0, -1, 0});
    std::vector<Eigen::Vector3i> triangles;
    triangles.push_back({0, 1, 2});
    triangles.push_back({0, 2, 3});
    triangles.push_back({0, 3, 4});
    triangles.push_back({0, 4, 1});
    mesh->SetVertices(vertices);
    mesh->SetTriangles(triangles);

    mesh = mesh->FilterSmoothLaplacian(1, 0.5);
    std::vector<Eigen::Vector3f> ref1;
    ref1.push_back({0, 0, 0});
    ref1.push_back({0.5, 0, 0});
    ref1.push_back({0, 0.5, 0});
    ref1.push_back({-0.5, 0, 0});
    ref1.push_back({0, -0.5, 0});
    ExpectEQ(mesh->GetVertices(), ref1);

    mesh = mesh->FilterSmoothLaplacian(10, 0.5);
    std::vector<Eigen::Vector3f> ref2;
    ref2.push_back({0, 0, 0});
    ref2.push_back({0.000488, 0, 0});
    ref2.push_back({0, 0.000488, 0});
    ref2.push_back({-0.000488, 0, 0});
    ref2.push_back({0, -0.000488, 0});
    ExpectEQ(mesh->GetVertices(), ref2);
}

TEST(TriangleMesh, FilterSmoothTaubin) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    std::vector<Eigen::Vector3f> vertices;
    vertices.push_back({0, 0, 0});
    vertices.push_back({1, 0, 0});
    vertices.push_back({0, 1, 0});
    vertices.push_back({-1, 0, 0});
    vertices.push_back({0, -1, 0});
    std::vector<Eigen::Vector3i> triangles;
    triangles.push_back({0, 1, 2});
    triangles.push_back({0, 2, 3});
    triangles.push_back({0, 3, 4});
    triangles.push_back({0, 4, 1});
    mesh->SetVertices(vertices);
    mesh->SetTriangles(triangles);

    mesh = mesh->FilterSmoothTaubin(1, 0.5, -0.53);
    std::vector<Eigen::Vector3f> ref1;
    ref1.push_back({0, 0, 0});
    ref1.push_back({0.765, 0, 0});
    ref1.push_back({0, 0.765, 0});
    ref1.push_back({-0.765, 0, 0});
    ref1.push_back({0, -0.765, 0});
    ExpectEQ(mesh->GetVertices(), ref1);

    mesh = mesh->FilterSmoothTaubin(10, 0.5, -0.53);
    std::vector<Eigen::Vector3f> ref2;
    ref2.push_back({0, 0, 0});
    ref2.push_back({0.052514, 0, 0});
    ref2.push_back({0, 0.052514, 0});
    ref2.push_back({-0.052514, 0, 0});
    ref2.push_back({0, -0.052514, 0});
    ExpectEQ(mesh->GetVertices(), ref2);
}

TEST(TriangleMesh, HasVertices) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertices());

    std::vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);

    EXPECT_TRUE(tm.HasVertices());
}

TEST(TriangleMesh, HasTriangles) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasTriangles());

    std::vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);
    std::vector<Vector3i> triangles(size);
    tm.SetTriangles(triangles);

    EXPECT_TRUE(tm.HasTriangles());
}

TEST(TriangleMesh, HasVertexNormals) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertexNormals());

    std::vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);
    std::vector<Vector3f> vertex_normals(size);
    tm.SetVertexNormals(vertex_normals);

    EXPECT_TRUE(tm.HasVertexNormals());
}

TEST(TriangleMesh, HasVertexColors) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertexColors());

    std::vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);
    std::vector<Vector3f> vertex_colors(size);
    tm.SetVertexColors(vertex_colors);

    EXPECT_TRUE(tm.HasVertexColors());
}

TEST(TriangleMesh, HasTriangleNormals) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasTriangleNormals());

    std::vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);
    std::vector<Vector3i> triangles(size);
    tm.SetTriangles(triangles);
    std::vector<Vector3f> triangle_normals(size);
    tm.SetTriangleNormals(triangle_normals);

    EXPECT_TRUE(tm.HasTriangleNormals());
}

TEST(TriangleMesh, NormalizeNormals) {
    Vector3f ref_vertex_normals_raw[] = {
            {0.692861, 0.323767, 0.644296}, {0.650010, 0.742869, 0.160101},
            {0.379563, 0.870761, 0.312581}, {0.575046, 0.493479, 0.652534},
            {0.320665, 0.448241, 0.834418}, {0.691127, 0.480526, 0.539850},
            {0.227557, 0.973437, 0.025284}, {0.281666, 0.156994, 0.946582},
            {0.341869, 0.894118, 0.289273}, {0.103335, 0.972118, 0.210498},
            {0.441745, 0.723783, 0.530094}, {0.336903, 0.727710, 0.597441},
            {0.434917, 0.862876, 0.257471}, {0.636619, 0.435239, 0.636619},
            {0.393717, 0.876213, 0.277918}, {0.275051, 0.633543, 0.723167},
            {0.061340, 0.873191, 0.483503}, {0.118504, 0.276510, 0.953677},
            {0.930383, 0.360677, 0.065578}, {0.042660, 0.989719, 0.136513},
            {0.175031, 0.720545, 0.670953}, {0.816905, 0.253392, 0.518130},
            {0.377967, 0.767871, 0.517219}, {0.782281, 0.621223, 0.046017},
            {0.314385, 0.671253, 0.671253}};
    std::vector<Vector3f> ref_vertex_normals(25);
    for (int i = 0; i < 25; ++i)
        ref_vertex_normals[i] = ref_vertex_normals_raw[i];

    Vector3f ref_triangle_normals_raw[] = {
            {0.331843, 0.660368, 0.673642}, {0.920309, 0.198342, 0.337182},
            {0.778098, 0.279317, 0.562624}, {0.547237, 0.723619, 0.420604},
            {0.360898, 0.671826, 0.646841}, {0.657733, 0.738934, 0.146163},
            {0.929450, 0.024142, 0.368159}, {0.160811, 0.969595, 0.184460},
            {0.922633, 0.298499, 0.244226}, {0.874092, 0.189272, 0.447370},
            {0.776061, 0.568382, 0.273261}, {0.663812, 0.544981, 0.512200},
            {0.763905, 0.227940, 0.603732}, {0.518555, 0.758483, 0.394721},
            {0.892885, 0.283206, 0.350074}, {0.657978, 0.751058, 0.054564},
            {0.872328, 0.483025, 0.075698}, {0.170605, 0.588415, 0.790356},
            {0.982336, 0.178607, 0.055815}, {0.881626, 0.121604, 0.456013},
            {0.616413, 0.573987, 0.539049}, {0.372896, 0.762489, 0.528733},
            {0.669715, 0.451103, 0.589905}, {0.771164, 0.057123, 0.634068},
            {0.620625, 0.620625, 0.479217}};
    std::vector<Vector3f> ref_triangle_normals(25);
    for (int i = 0; i < 25; ++i)
        ref_triangle_normals[i] = ref_triangle_normals_raw[i];

    int size = 25;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(10.0, 10.0, 10.0);

    geometry::TriangleMesh tm;

    std::vector<Vector3f> vertex_normals(size);
    std::vector<Vector3f> triangle_normals(size);
    Rand(vertex_normals, dmin, dmax, 0);
    Rand(triangle_normals, dmin, dmax, 1);
    tm.SetVertexNormals(vertex_normals);
    tm.SetTriangleNormals(triangle_normals);

    tm.NormalizeNormals();

    ExpectEQ(ref_vertex_normals, tm.GetVertexNormals());
    ExpectEQ(ref_triangle_normals, tm.GetTriangleNormals());
}

TEST(TriangleMesh, PaintUniformColor) {
    int size = 25;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(10.0, 10.0, 10.0);

    geometry::TriangleMesh tm;

    std::vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);
    std::vector<Vector3f> vertex_colors(size);
    tm.SetVertexColors(vertex_colors);

    Vector3f color(233. / 255., 171. / 255., 53.0 / 255.);
    tm.PaintUniformColor(color);

    vertex_colors = tm.GetVertexColors();
    for (size_t i = 0; i < tm.vertex_colors_.size(); i++)
        ExpectEQ(color, vertex_colors[i]);
}

TEST(TriangleMesh, CreateMeshSphere) {
    Vector3f ref_vertices_raw[] = {{0.000000, 0.000000, 1.000000},
                                   {0.000000, 0.000000, -1.000000},
                                   {0.587785, 0.000000, 0.809017},
                                   {0.475528, 0.345492, 0.809017},
                                   {0.181636, 0.559017, 0.809017},
                                   {-0.181636, 0.559017, 0.809017},
                                   {-0.475528, 0.345492, 0.809017},
                                   {-0.587785, 0.000000, 0.809017},
                                   {-0.475528, -0.345492, 0.809017},
                                   {-0.181636, -0.559017, 0.809017},
                                   {0.181636, -0.559017, 0.809017},
                                   {0.475528, -0.345492, 0.809017},
                                   {0.951057, 0.000000, 0.309017},
                                   {0.769421, 0.559017, 0.309017},
                                   {0.293893, 0.904508, 0.309017},
                                   {-0.293893, 0.904508, 0.309017},
                                   {-0.769421, 0.559017, 0.309017},
                                   {-0.951057, 0.000000, 0.309017},
                                   {-0.769421, -0.559017, 0.309017},
                                   {-0.293893, -0.904508, 0.309017},
                                   {0.293893, -0.904508, 0.309017},
                                   {0.769421, -0.559017, 0.309017},
                                   {0.951057, 0.000000, -0.309017},
                                   {0.769421, 0.559017, -0.309017},
                                   {0.293893, 0.904508, -0.309017},
                                   {-0.293893, 0.904508, -0.309017},
                                   {-0.769421, 0.559017, -0.309017},
                                   {-0.951057, 0.000000, -0.309017},
                                   {-0.769421, -0.559017, -0.309017},
                                   {-0.293893, -0.904508, -0.309017},
                                   {0.293893, -0.904508, -0.309017},
                                   {0.769421, -0.559017, -0.309017},
                                   {0.587785, 0.000000, -0.809017},
                                   {0.475528, 0.345492, -0.809017},
                                   {0.181636, 0.559017, -0.809017},
                                   {-0.181636, 0.559017, -0.809017},
                                   {-0.475528, 0.345492, -0.809017},
                                   {-0.587785, 0.000000, -0.809017},
                                   {-0.475528, -0.345492, -0.809017},
                                   {-0.181636, -0.559017, -0.809017},
                                   {0.181636, -0.559017, -0.809017},
                                   {0.475528, -0.345492, -0.809017}};
    std::vector<Vector3f> ref_vertices(42);
    for (int i = 0; i < 42; ++i) ref_vertices[i] = ref_vertices_raw[i];

    Vector3i ref_triangles_raw[] = {
            {0, 2, 3},    {1, 33, 32},  {0, 3, 4},    {1, 34, 33},
            {0, 4, 5},    {1, 35, 34},  {0, 5, 6},    {1, 36, 35},
            {0, 6, 7},    {1, 37, 36},  {0, 7, 8},    {1, 38, 37},
            {0, 8, 9},    {1, 39, 38},  {0, 9, 10},   {1, 40, 39},
            {0, 10, 11},  {1, 41, 40},  {0, 11, 2},   {1, 32, 41},
            {12, 3, 2},   {12, 13, 3},  {13, 4, 3},   {13, 14, 4},
            {14, 5, 4},   {14, 15, 5},  {15, 6, 5},   {15, 16, 6},
            {16, 7, 6},   {16, 17, 7},  {17, 8, 7},   {17, 18, 8},
            {18, 9, 8},   {18, 19, 9},  {19, 10, 9},  {19, 20, 10},
            {20, 11, 10}, {20, 21, 11}, {21, 2, 11},  {21, 12, 2},
            {22, 13, 12}, {22, 23, 13}, {23, 14, 13}, {23, 24, 14},
            {24, 15, 14}, {24, 25, 15}, {25, 16, 15}, {25, 26, 16},
            {26, 17, 16}, {26, 27, 17}, {27, 18, 17}, {27, 28, 18},
            {28, 19, 18}, {28, 29, 19}, {29, 20, 19}, {29, 30, 20},
            {30, 21, 20}, {30, 31, 21}, {31, 12, 21}, {31, 22, 12},
            {32, 23, 22}, {32, 33, 23}, {33, 24, 23}, {33, 34, 24},
            {34, 25, 24}, {34, 35, 25}, {35, 26, 25}, {35, 36, 26},
            {36, 27, 26}, {36, 37, 27}, {37, 28, 27}, {37, 38, 28},
            {38, 29, 28}, {38, 39, 29}, {39, 30, 29}, {39, 40, 30},
            {40, 31, 30}, {40, 41, 31}, {41, 22, 31}, {41, 32, 22}};
    std::vector<Vector3i> ref_triangles(80);
    for (int i = 0; i < 80; ++i) ref_triangles[i] = ref_triangles_raw[i];

    auto output_tm = geometry::TriangleMesh::CreateSphere(1.0, 5);

    ExpectEQ(ref_vertices, output_tm->GetVertices());
    ExpectEQ(ref_triangles, output_tm->GetTriangles());
}

TEST(TriangleMesh, CreateMeshCylinder) {
    Vector3f ref_vertices_raw[] = {{0.000000, 0.000000, 1.000000},
                                   {0.000000, 0.000000, -1.000000},
                                   {1.000000, 0.000000, 1.000000},
                                   {0.309017, 0.951057, 1.000000},
                                   {-0.809017, 0.587785, 1.000000},
                                   {-0.809017, -0.587785, 1.000000},
                                   {0.309017, -0.951057, 1.000000},
                                   {1.000000, 0.000000, 0.500000},
                                   {0.309017, 0.951057, 0.500000},
                                   {-0.809017, 0.587785, 0.500000},
                                   {-0.809017, -0.587785, 0.500000},
                                   {0.309017, -0.951057, 0.500000},
                                   {1.000000, 0.000000, 0.000000},
                                   {0.309017, 0.951057, 0.000000},
                                   {-0.809017, 0.587785, 0.000000},
                                   {-0.809017, -0.587785, 0.000000},
                                   {0.309017, -0.951057, 0.000000},
                                   {1.000000, 0.000000, -0.500000},
                                   {0.309017, 0.951057, -0.500000},
                                   {-0.809017, 0.587785, -0.500000},
                                   {-0.809017, -0.587785, -0.500000},
                                   {0.309017, -0.951057, -0.500000},
                                   {1.000000, 0.000000, -1.000000},
                                   {0.309017, 0.951057, -1.000000},
                                   {-0.809017, 0.587785, -1.000000},
                                   {-0.809017, -0.587785, -1.000000},
                                   {0.309017, -0.951057, -1.000000}};
    std::vector<Vector3f> ref_vertices(27);
    for (int i = 0; i < 27; ++i) ref_vertices[i] = ref_vertices_raw[i];

    Vector3i ref_triangles_raw[] = {
            {0, 2, 3},    {1, 23, 22},  {0, 3, 4},    {1, 24, 23},
            {0, 4, 5},    {1, 25, 24},  {0, 5, 6},    {1, 26, 25},
            {0, 6, 2},    {1, 22, 26},  {7, 3, 2},    {7, 8, 3},
            {8, 4, 3},    {8, 9, 4},    {9, 5, 4},    {9, 10, 5},
            {10, 6, 5},   {10, 11, 6},  {11, 2, 6},   {11, 7, 2},
            {12, 8, 7},   {12, 13, 8},  {13, 9, 8},   {13, 14, 9},
            {14, 10, 9},  {14, 15, 10}, {15, 11, 10}, {15, 16, 11},
            {16, 7, 11},  {16, 12, 7},  {17, 13, 12}, {17, 18, 13},
            {18, 14, 13}, {18, 19, 14}, {19, 15, 14}, {19, 20, 15},
            {20, 16, 15}, {20, 21, 16}, {21, 12, 16}, {21, 17, 12},
            {22, 18, 17}, {22, 23, 18}, {23, 19, 18}, {23, 24, 19},
            {24, 20, 19}, {24, 25, 20}, {25, 21, 20}, {25, 26, 21},
            {26, 17, 21}, {26, 22, 17}};
    std::vector<Vector3i> ref_triangles(50);
    for (int i = 0; i < 50; ++i) ref_triangles[i] = ref_triangles_raw[i];

    auto output_tm = geometry::TriangleMesh::CreateCylinder(1.0, 2.0, 5);

    ExpectEQ(ref_vertices, output_tm->GetVertices());
    ExpectEQ(ref_triangles, output_tm->GetTriangles());
}

TEST(TriangleMesh, CreateMeshCone) {
    Vector3f ref_vertices_raw[] = {
            {0.000000, 0.000000, 0.000000},  {0.000000, 0.000000, 2.000000},
            {1.000000, 0.000000, 0.000000},  {0.309017, 0.951057, 0.000000},
            {-0.809017, 0.587785, 0.000000}, {-0.809017, -0.587785, 0.000000},
            {0.309017, -0.951057, 0.000000}};
    std::vector<Vector3f> ref_vertices(7);
    for (int i = 0; i < 7; ++i) ref_vertices[i] = ref_vertices_raw[i];

    Vector3i ref_triangles_raw[] = {{0, 3, 2}, {1, 2, 3}, {0, 4, 3}, {1, 3, 4},
                                    {0, 5, 4}, {1, 4, 5}, {0, 6, 5}, {1, 5, 6},
                                    {0, 2, 6}, {1, 6, 2}};
    std::vector<Vector3i> ref_triangles(10);
    for (int i = 0; i < 10; ++i) ref_triangles[i] = ref_triangles_raw[i];

    auto output_tm = geometry::TriangleMesh::CreateCone(1.0, 2.0, 5);

    ExpectEQ(ref_vertices, output_tm->GetVertices());
    ExpectEQ(ref_triangles, output_tm->GetTriangles());
}

TEST(TriangleMesh, CreateMeshArrow) {
    Vector3f ref_vertices_raw[] = {
            {0.000000, 0.000000, 2.000000},   {0.000000, 0.000000, 0.000000},
            {1.000000, 0.000000, 2.000000},   {0.309017, 0.951057, 2.000000},
            {-0.809017, 0.587785, 2.000000},  {-0.809017, -0.587785, 2.000000},
            {0.309017, -0.951057, 2.000000},  {1.000000, 0.000000, 1.500000},
            {0.309017, 0.951057, 1.500000},   {-0.809017, 0.587785, 1.500000},
            {-0.809017, -0.587785, 1.500000}, {0.309017, -0.951057, 1.500000},
            {1.000000, 0.000000, 1.000000},   {0.309017, 0.951057, 1.000000},
            {-0.809017, 0.587785, 1.000000},  {-0.809017, -0.587785, 1.000000},
            {0.309017, -0.951057, 1.000000},  {1.000000, 0.000000, 0.500000},
            {0.309017, 0.951057, 0.500000},   {-0.809017, 0.587785, 0.500000},
            {-0.809017, -0.587785, 0.500000}, {0.309017, -0.951057, 0.500000},
            {1.000000, 0.000000, 0.000000},   {0.309017, 0.951057, 0.000000},
            {-0.809017, 0.587785, 0.000000},  {-0.809017, -0.587785, 0.000000},
            {0.309017, -0.951057, 0.000000},  {0.000000, 0.000000, 2.000000},
            {0.000000, 0.000000, 3.000000},   {1.500000, 0.000000, 2.000000},
            {0.463525, 1.426585, 2.000000},   {-1.213525, 0.881678, 2.000000},
            {-1.213525, -0.881678, 2.000000}, {0.463525, -1.426585, 2.000000}};
    std::vector<Vector3f> ref_vertices(34);
    for (int i = 0; i < 34; ++i) ref_vertices[i] = ref_vertices_raw[i];

    Vector3i ref_triangles_raw[] = {
            {0, 2, 3},    {1, 23, 22},  {0, 3, 4},    {1, 24, 23},
            {0, 4, 5},    {1, 25, 24},  {0, 5, 6},    {1, 26, 25},
            {0, 6, 2},    {1, 22, 26},  {7, 3, 2},    {7, 8, 3},
            {8, 4, 3},    {8, 9, 4},    {9, 5, 4},    {9, 10, 5},
            {10, 6, 5},   {10, 11, 6},  {11, 2, 6},   {11, 7, 2},
            {12, 8, 7},   {12, 13, 8},  {13, 9, 8},   {13, 14, 9},
            {14, 10, 9},  {14, 15, 10}, {15, 11, 10}, {15, 16, 11},
            {16, 7, 11},  {16, 12, 7},  {17, 13, 12}, {17, 18, 13},
            {18, 14, 13}, {18, 19, 14}, {19, 15, 14}, {19, 20, 15},
            {20, 16, 15}, {20, 21, 16}, {21, 12, 16}, {21, 17, 12},
            {22, 18, 17}, {22, 23, 18}, {23, 19, 18}, {23, 24, 19},
            {24, 20, 19}, {24, 25, 20}, {25, 21, 20}, {25, 26, 21},
            {26, 17, 21}, {26, 22, 17}, {27, 30, 29}, {28, 29, 30},
            {27, 31, 30}, {28, 30, 31}, {27, 32, 31}, {28, 31, 32},
            {27, 33, 32}, {28, 32, 33}, {27, 29, 33}, {28, 33, 29}};
    std::vector<Vector3i> ref_triangles(60);
    for (int i = 0; i < 60; ++i) ref_triangles[i] = ref_triangles_raw[i];

    auto output_tm = geometry::TriangleMesh::CreateArrow(1.0, 1.5, 2.0, 1.0, 5);

    ExpectEQ(ref_vertices, output_tm->GetVertices());
    ExpectEQ(ref_triangles, output_tm->GetTriangles());
}
