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
#include "cupoch/registration/kabsch.h"

#include "cupoch/geometry/pointcloud.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

namespace {
float deg_to_rad(float deg) { return deg / 180.0 * M_PI; }
}  // namespace

TEST(Kabsch, Kabsch) {
    const size_t size = 20;
    Vector3f vmin(0.0, 0.0, 0.0);
    Vector3f vmax(1000.0, 1000.0, 1000.0);
    std::vector<Vector3f> points(size);
    Rand(points, vmin, vmax, 0);
    geometry::PointCloud source;
    source.SetPoints(points);
    const float rad = deg_to_rad(30.0f);
    const Matrix4f ref_tf = (Matrix4f() << std::cos(rad), -std::sin(rad), 0.0,
                             0.0, std::sin(rad), std::cos(rad), 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                                    .finished();

    geometry::PointCloud target = source;
    target.Transform(ref_tf);
    const Matrix4f res = registration::Kabsch(source.GetPoints(), target.GetPoints());
    std::cout << ref_tf << std::endl;
    std::cout << res << std::endl;
    EXPECT_TRUE(res.isApprox(ref_tf, 1.0e-3));
}