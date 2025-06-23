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
#include "cupoch/geometry/rgbdimage.h"
#include "cupoch/odometry/rgbdodometry_jacobian.h"
#include "tests/odometry/odometry_tools.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace odometry_tools;
using namespace cupoch;
using namespace unit_test;

TEST(RGBDOdometryJacobianFromColorTerm, ComputeJacobianAndResidual) {
    std::vector<Vector6f> ref_J_r(10);
    ref_J_r[0] << -1.208103, 0.621106, -0.040830, 0.173142, 0.260220, -1.164557;
    ref_J_r[1] << -0.338017, 0.140257, 0.019732, 0.030357, 0.128839, -0.395772;
    ref_J_r[2] << -0.235842, 0.122008, 0.029948, 0.037260, 0.119792, -0.194611;
    ref_J_r[3] << -0.222063, 0.118091, -0.018617, 0.096335, 0.144784, -0.230677;
    ref_J_r[4] << -0.127762, 0.197381, 0.104905, 0.072993, 0.146487, -0.186723;
    ref_J_r[5] << -0.012070, 0.033963, -0.004087, 0.019158, 0.004083, -0.022654;
    ref_J_r[6] << -0.047053, 0.049144, -0.027889, 0.040064, 0.010937, -0.048321;
    ref_J_r[7] << -0.338017, 0.140257, 0.019732, 0.030357, 0.128839, -0.395772;
    ref_J_r[8] << -2.080471, 1.779082, 0.191770, 0.116250, 0.373750, -2.206175;
    ref_J_r[9] << -0.015476, 0.054573, -0.002288, 0.027828, 0.005931, -0.046776;

    float ref_r_raw[] = {0.419608,  -0.360784, 0.274510,  0.564706, 0.835294,
                         -0.352941, -0.545098, -0.360784, 0.121569, -0.094118};
    std::vector<float> ref_r;
    for (int i = 0; i < 10; ++i) ref_r.push_back(ref_r_raw[i]);

    int width = 10;
    int height = 10;
    // int num_of_channels = 1;
    // int bytes_per_channel = 4;

    auto srcColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    auto srcDepth = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 0);

    auto tgtColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    auto tgtDepth = GenerateImage(width, height, 1, 4, 1.0f, 2.0f, 0);

    auto dxColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);
    auto dyColor = GenerateImage(width, height, 1, 4, 0.0f, 1.0f, 1);

    ShiftLeft(tgtColor, 10);
    ShiftUp(tgtColor, 5);

    ShiftLeft(dxColor, 10);
    ShiftUp(dyColor, 5);

    geometry::RGBDImage source(*srcColor, *srcDepth);
    geometry::RGBDImage target(*tgtColor, *tgtDepth);
    auto source_xyz = GenerateImage(width, height, 3, 4, 0.0f, 1.0f, 0);
    geometry::RGBDImage target_dx(*dxColor, *tgtDepth);
    geometry::RGBDImage target_dy(*dyColor, *tgtDepth);

    Matrix3f intrinsic = Matrix3f::Zero();
    intrinsic(0, 0) = 0.5;
    intrinsic(1, 1) = 0.65;
    intrinsic(0, 2) = 0.75;
    intrinsic(1, 2) = 0.35;

    Matrix4f extrinsic = Matrix4f::Zero();
    extrinsic(0, 0) = 1.0;
    extrinsic(1, 1) = 1.0;
    extrinsic(2, 2) = 1.0;

    int rows = height;
    std::vector<Vector4i> corresps(rows);
    Rand(corresps, 0, 3, 0);

    odometry::RGBDOdometryJacobianFromColorTerm jacobian_method;

    for (int row = 0; row < rows; row++) {
        Vector6f J_r[2];
        float r[2];

        jacobian_method.ComputeJacobianAndResidual(
                row, J_r, r,
                thrust::raw_pointer_cast(source.color_.GetData().data()),
                thrust::raw_pointer_cast(source.depth_.GetData().data()),
                thrust::raw_pointer_cast(target.color_.GetData().data()),
                thrust::raw_pointer_cast(target.depth_.GetData().data()),
                thrust::raw_pointer_cast(source_xyz->GetData().data()),
                thrust::raw_pointer_cast(target_dx.color_.GetData().data()),
                thrust::raw_pointer_cast(target_dx.depth_.GetData().data()),
                thrust::raw_pointer_cast(target_dy.color_.GetData().data()),
                thrust::raw_pointer_cast(target_dy.depth_.GetData().data()),
                width, intrinsic, extrinsic,
                thrust::raw_pointer_cast(corresps.data()));

        EXPECT_NEAR(ref_r[row], r[0], THRESHOLD_1E_4);
        ExpectEQ(ref_J_r[row], J_r[0]);
    }
}