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
#include "cupoch/knn/lbvh_knn.h"

#include <thrust/remove.h>
#include <thrust/sort.h>

#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/geometry_utils.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

namespace {

struct is_minus_one_functor {
    bool operator()(int x) const { return (x == -1); }
};

struct is_inf_functor {
    bool operator()(float x) const { return std::isinf(x); }
};

}  // namespace

TEST(LinearBoundingVolumeHierarchyKNN, SearchKNN) {
    std::vector<int> ref_indices;
    int indices0[] = {27, 48, 4,  77, 90, 7,  54, 17, 76, 38,
                      39, 60, 15, 84, 11, 57, 3,  32, 99, 36,
                      52, 40, 26, 59, 22, 97, 20, 42, 73, 24};
    for (int i = 0; i < 30; ++i) ref_indices.push_back(indices0[i]);

    std::vector<float> ref_distance2;
    float distances0[] = {
            0.000000,  4.684353,  4.996539,  9.191849,  10.034604, 10.466745,
            10.649751, 11.434066, 12.089195, 13.345638, 13.696270, 14.016148,
            16.851978, 17.073435, 18.254518, 20.019994, 21.496347, 23.077277,
            23.692427, 23.809303, 24.104578, 25.005770, 26.952710, 27.487888,
            27.998463, 28.262975, 28.581313, 28.816608, 31.603230, 31.610916};
    for (int i = 0; i < 30; ++i) ref_distance2.push_back(distances0[i]);

    int size = 100;

    geometry::PointCloud pc;

    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(10.0, 10.0, 10.0);

    std::vector<Eigen::Vector3f> points(size);
    Rand(points, vmin, vmax, 0);
    pc.SetPoints(points);

    knn::LinearBoundingVolumeHierarchyKNN kdtree(geometry::ConvertVector3fStdVector(pc));

    Eigen::Vector3f query = {1.647059, 4.392157, 8.784314};
    std::vector<unsigned int> indices;
    std::vector<float> distance2;

    int result = kdtree.SearchNN(query, std::numeric_limits<float>::max(), indices, distance2);

    EXPECT_EQ(result, 1);

    EXPECT_EQ(ref_indices[0], indices[0]);
    EXPECT_NEAR(ref_distance2[0], distance2[0], 1.0e-9);
}
