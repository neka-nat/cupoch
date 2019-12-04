#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/geometry/pointcloud.h"
#include "tests/test_utility/unit_test.h"
#include <thrust/sort.h>
#include <thrust/remove.h>

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

namespace {

struct is_minus_one_functor {
    bool operator() (int x) const {
        return (x == -1);
    }
};

struct is_inf_functor {
    bool operator() (float x) const {
        return std::isinf(x);
    }
};

}

TEST(KDTreeFlann, SearchKNN) {
    thrust::host_vector<int> ref_indices;
    int indices0[] = {27, 48, 4,  77, 90, 7,  54, 17, 76, 38,
                      39, 60, 15, 84, 11, 57, 3,  32, 99, 36,
                      52, 40, 26, 59, 22, 97, 20, 42, 73, 24};
    for (int i = 0; i < 30; ++i) ref_indices.push_back(indices0[i]);

    thrust::host_vector<float> ref_distance2;
    float distances0[] = {0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578, 25.005770, 26.952710, 27.487888,
            27.998463, 28.262975, 28.581313, 28.816608, 31.603230, 31.610916};
    for (int i = 0; i < 30; ++i) ref_distance2.push_back(distances0[i]);

    int size = 100;

    geometry::PointCloud pc;

    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(10.0, 10.0, 10.0);

    thrust::host_vector<Eigen::Vector3f> points(size);
    Rand(points, vmin, vmax, 0);
    pc.SetPoints(points);

    geometry::KDTreeFlann kdtree(pc);

    Eigen::Vector3f query = {1.647059, 4.392157, 8.784314};
    int knn = 30;
    thrust::host_vector<int> indices;
    thrust::host_vector<float> distance2;

    int result = kdtree.SearchKNN(query, knn, indices, distance2);

    EXPECT_EQ(result, 30);

    thrust::sort(ref_indices.begin(), ref_indices.end());
    thrust::sort(indices.begin(), indices.end());
    thrust::sort(ref_distance2.begin(), ref_distance2.end());
    thrust::sort(distance2.begin(), distance2.end());
    ExpectEQ(ref_indices, indices);
    ExpectEQ(ref_distance2, distance2);
}

TEST(KDTreeFlann, SearchRadius) {
    thrust::host_vector<int> ref_indices;
    int indices0[] = {27, 48, 4,  77, 90, 7, 54, 17, 76, 38, 39,
                      60, 15, 84, 11, 57, 3, 32, 99, 36, 52};
    for (int i = 0; i < 21; ++i) ref_indices.push_back(indices0[i]);

    thrust::host_vector<float> ref_distance2;
    float distances0[] = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578};
    for (int i = 0; i < 21; ++i) ref_distance2.push_back(distances0[i]);

    int size = 100;

    geometry::PointCloud pc;

    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(10.0, 10.0, 10.0);

    thrust::host_vector<Eigen::Vector3f> points(size);
    Rand(points, vmin, vmax, 0);
    pc.SetPoints(points);

    geometry::KDTreeFlann kdtree(pc);

    Eigen::Vector3f query = {1.647059, 4.392157, 8.784314};
    float radius = 5.0;
    thrust::host_vector<int> indices;
    thrust::host_vector<float> distance2;

    int result =
            kdtree.SearchRadius<Vector3f>(query, radius, indices, distance2);

    EXPECT_EQ(result, 21);

    thrust::remove_if(indices.begin(), indices.end(), is_minus_one_functor());
    thrust::remove_if(distance2.begin(), distance2.end(), is_inf_functor());
    indices.resize(result);
    distance2.resize(result);
    thrust::sort(ref_indices.begin(), ref_indices.end());
    thrust::sort(indices.begin(), indices.end());
    thrust::sort(ref_distance2.begin(), ref_distance2.end());
    thrust::sort(distance2.begin(), distance2.end());
    ExpectEQ(ref_indices, indices);
    ExpectEQ(ref_distance2, distance2);
}

TEST(KDTreeFlann, SearchHybrid) {
    thrust::host_vector<int> ref_indices;
    int indices0[] = {27, 48, 4,  77, 90, 7,  54, 17,
                     76, 38, 39, 60, 15, 84, 11};
    for (int i = 0; i < 15; ++i) ref_indices.push_back(indices0[i]);

    thrust::host_vector<float> ref_distance2;
    float distances0[] = {0.000000,  4.684353,  4.996539,  9.191849,
                          10.034604, 10.466745, 10.649751, 11.434066,
                          12.089195, 13.345638, 13.696270, 14.016148,
                          16.851978, 17.073435, 18.254518};
    for (int i = 0; i < 15; ++i) ref_distance2.push_back(distances0[i]);

    int size = 100;

    geometry::PointCloud pc;

    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(10.0, 10.0, 10.0);

    thrust::host_vector<Eigen::Vector3f> points(size);
    Rand(points, vmin, vmax, 0);
    pc.SetPoints(points);

    geometry::KDTreeFlann kdtree(pc);

    Eigen::Vector3f query = {1.647059, 4.392157, 8.784314};
    int max_nn = 15;
    float radius = 5.0;
    thrust::host_vector<int> indices;
    thrust::host_vector<float> distance2;

    int result = kdtree.SearchHybrid<Vector3f>(query, radius, max_nn, indices,
                                                 distance2);

    EXPECT_EQ(result, 15);

    thrust::sort(ref_indices.begin(), ref_indices.end());
    thrust::sort(indices.begin(), indices.end());
    thrust::sort(ref_distance2.begin(), ref_distance2.end());
    thrust::sort(distance2.begin(), distance2.end());
    ExpectEQ(ref_indices, indices);
    ExpectEQ(ref_distance2, distance2);
}