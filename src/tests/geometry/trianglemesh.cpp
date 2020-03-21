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

    thrust::host_vector<Vector3f> vertices(size);
    thrust::host_vector<Vector3f> vertex_normals(size);
    thrust::host_vector<Vector3f> vertex_colors(size);
    thrust::host_vector<Vector3i> triangles(size);
    thrust::host_vector<Vector3f> triangle_normals(size);

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

    thrust::host_vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);

    EXPECT_FALSE(tm.IsEmpty());
}

TEST(TriangleMesh, GetMinBound) {
    int size = 100;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(1000.0, 1000.0, 1000.0);

    geometry::TriangleMesh tm;

    thrust::host_vector<Vector3f> vertices(size);
    Rand(vertices, dmin, dmax, 0);
    tm.SetVertices(vertices);

    ExpectEQ(Vector3f(19.607843, 0.0, 0.0), tm.GetMinBound());
}

TEST(TriangleMesh, GetMaxBound) {
    int size = 100;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(1000.0, 1000.0, 1000.0);

    geometry::TriangleMesh tm;

    thrust::host_vector<Vector3f> vertices(size);
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

    thrust::host_vector<Vector3f> vertices0(size);
    thrust::host_vector<Vector3f> vertex_normals0(size);
    thrust::host_vector<Vector3f> vertex_colors0(size);
    thrust::host_vector<Vector3i> triangles0(size);
    thrust::host_vector<Vector3f> triangle_normals0(size);
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

    thrust::host_vector<Vector3f> vertices1(size);
    thrust::host_vector<Vector3f> vertex_normals1(size);
    thrust::host_vector<Vector3f> vertex_colors1(size);
    thrust::host_vector<Vector3i> triangles1(size);
    thrust::host_vector<Vector3f> triangle_normals1(size);
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

    thrust::host_vector<Vector3f> vertices = tm.GetVertices();
    thrust::host_vector<Vector3f> vertex_normals = tm.GetVertexNormals();
    thrust::host_vector<Vector3f> vertex_colors = tm.GetVertexColors();
    thrust::host_vector<Vector3i> triangles = tm.GetTriangles();
    thrust::host_vector<Vector3f> triangle_normals = tm.GetTriangleNormals();

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
        ExpectEQ(Vector3i(triangles1[i](0, 0) + size,
                          triangles1[i](1, 0) + size,
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

    thrust::host_vector<Vector3f> vertices0(size);
    thrust::host_vector<Vector3f> vertex_normals0(size);
    thrust::host_vector<Vector3f> vertex_colors0(size);
    thrust::host_vector<Vector3i> triangles0(size);
    thrust::host_vector<Vector3f> triangle_normals0(size);
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

    thrust::host_vector<Vector3f> vertices1(size);
    thrust::host_vector<Vector3f> vertex_normals1(size);
    thrust::host_vector<Vector3f> vertex_colors1(size);
    thrust::host_vector<Vector3i> triangles1(size);
    thrust::host_vector<Vector3f> triangle_normals1(size);
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

    thrust::host_vector<Vector3f> vertices = tm.GetVertices();
    thrust::host_vector<Vector3f> vertex_normals = tm.GetVertexNormals();
    thrust::host_vector<Vector3f> vertex_colors = tm.GetVertexColors();
    thrust::host_vector<Vector3i> triangles = tm.GetTriangles();
    thrust::host_vector<Vector3f> triangle_normals = tm.GetTriangleNormals();

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
        ExpectEQ(Vector3i(triangles1[i](0, 0) + size,
                          triangles1[i](1, 0) + size,
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
    thrust::host_vector<Vector3f> ref(25);
    for (int i = 0; i < 25; ++i) ref[i] = ref_raw[i];

    size_t size = 25;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(10.0, 10.0, 10.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    thrust::host_vector<Vector3f> vertices(size);
    Rand(vertices, dmin, dmax, 0);
    tm.SetVertices(vertices);

    thrust::host_vector<Vector3i> triangles;
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
    thrust::host_vector<Vector3f> ref(25);
    for (int i = 0; i < 25; ++i) ref[i] = ref_raw[i];

    size_t size = 25;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(10.0, 10.0, 10.0);

    Vector3i imin(0, 0, 0);
    Vector3i imax(size - 1, size - 1, size - 1);

    geometry::TriangleMesh tm;

    thrust::host_vector<Vector3f> vertices(size);
    Rand(vertices, dmin, dmax, 0);
    tm.SetVertices(vertices);

    thrust::host_vector<Vector3i> triangles;
    for (size_t i = 0; i < size; i++)
        triangles.push_back(Vector3i(i, (i + 1) % size, (i + 2) % size));
    tm.SetTriangles(triangles);

    tm.ComputeVertexNormals();

    ExpectEQ(ref, tm.GetVertexNormals());
}

TEST(TriangleMesh, HasVertices) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertices());

    thrust::host_vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);

    EXPECT_TRUE(tm.HasVertices());
}

TEST(TriangleMesh, HasTriangles) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasTriangles());

    thrust::host_vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);
    thrust::host_vector<Vector3i> triangles(size);
    tm.SetTriangles(triangles);

    EXPECT_TRUE(tm.HasTriangles());
}

TEST(TriangleMesh, HasVertexNormals) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertexNormals());

    thrust::host_vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);
    thrust::host_vector<Vector3f> vertex_normals(size);
    tm.SetVertexNormals(vertex_normals);

    EXPECT_TRUE(tm.HasVertexNormals());
}

TEST(TriangleMesh, HasVertexColors) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasVertexColors());

    thrust::host_vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);
    thrust::host_vector<Vector3f> vertex_colors(size);
    tm.SetVertexColors(vertex_colors);

    EXPECT_TRUE(tm.HasVertexColors());
}

TEST(TriangleMesh, HasTriangleNormals) {
    int size = 100;

    geometry::TriangleMesh tm;

    EXPECT_FALSE(tm.HasTriangleNormals());

    thrust::host_vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);
    thrust::host_vector<Vector3i> triangles(size);
    tm.SetTriangles(triangles);
    thrust::host_vector<Vector3f> triangle_normals(size);
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
    thrust::host_vector<Vector3f> ref_vertex_normals(25);
    for (int i = 0; i < 25; ++i) ref_vertex_normals[i] = ref_vertex_normals_raw[i];

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
    thrust::host_vector<Vector3f> ref_triangle_normals(25);
    for (int i = 0; i < 25; ++i) ref_triangle_normals[i] = ref_triangle_normals_raw[i];

    int size = 25;

    Vector3f dmin(0.0, 0.0, 0.0);
    Vector3f dmax(10.0, 10.0, 10.0);

    geometry::TriangleMesh tm;

    thrust::host_vector<Vector3f> vertex_normals(size);
    thrust::host_vector<Vector3f> triangle_normals(size);
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

    thrust::host_vector<Vector3f> vertices(size);
    tm.SetVertices(vertices);
    thrust::host_vector<Vector3f> vertex_colors(size);
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
    thrust::host_vector<Vector3f> ref_vertices(42);
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
    thrust::host_vector<Vector3i> ref_triangles(80);
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
    thrust::host_vector<Vector3f> ref_vertices(27);
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
    thrust::host_vector<Vector3i> ref_triangles(50);
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
    thrust::host_vector<Vector3f> ref_vertices(7);
    for (int i = 0; i < 7; ++i) ref_vertices[i] = ref_vertices_raw[i];

    Vector3i ref_triangles_raw[] = {
            {0, 3, 2}, {1, 2, 3}, {0, 4, 3}, {1, 3, 4}, {0, 5, 4},
            {1, 4, 5}, {0, 6, 5}, {1, 5, 6}, {0, 2, 6}, {1, 6, 2}};
    thrust::host_vector<Vector3i> ref_triangles(10);
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
    thrust::host_vector<Vector3f> ref_vertices(34);
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
    thrust::host_vector<Vector3i> ref_triangles(60);
    for (int i = 0; i < 60; ++i) ref_triangles[i] = ref_triangles_raw[i];

    auto output_tm = geometry::TriangleMesh::CreateArrow(1.0, 1.5, 2.0, 1.0, 5);

    ExpectEQ(ref_vertices, output_tm->GetVertices());
    ExpectEQ(ref_triangles, output_tm->GetTriangles());
}
