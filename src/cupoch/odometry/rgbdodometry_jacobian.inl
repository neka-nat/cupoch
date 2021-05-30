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
#include "cupoch/geometry/image.h"
#include "cupoch/geometry/rgbdimage.h"
#include "cupoch/odometry/rgbdodometry_jacobian.h"

#ifndef __CUDACC__
using std::isnan;
#endif

namespace cupoch {
namespace odometry {

namespace {

__device__ const float SOBEL_SCALE = 0.125;
__device__ const float LAMBDA_HYBRID_DEPTH = 0.968;

}  // unnamed namespace

__host__ __device__ inline void
RGBDOdometryJacobianFromColorTerm::ComputeJacobianAndResidual(
        int row,
        Eigen::Vector6f J_r[2],
        float r[2],
        const uint8_t *source_color,
        const uint8_t *source_depth,
        const uint8_t *target_color,
        const uint8_t *target_depth,
        const uint8_t *source_xyz,
        const uint8_t *target_dx_color,
        const uint8_t *target_dx_depth,
        const uint8_t *target_dy_color,
        const uint8_t *target_dy_depth,
        int width,
        const Eigen::Matrix3f &intrinsic,
        const Eigen::Matrix4f &extrinsic,
        const Eigen::Vector4i *corresps) const {
    Eigen::Matrix3f R = extrinsic.block<3, 3>(0, 0);
    Eigen::Vector3f t = extrinsic.block<3, 1>(0, 3);

    Eigen::Vector4i corresp = corresps[row];
    int u_s = corresp(0);
    int v_s = corresp(1);
    int u_t = corresp(2);
    int v_t = corresp(3);
    float diff = *geometry::PointerAt<float>(target_color, width, u_t, v_t) -
                 *geometry::PointerAt<float>(source_color, width, u_s, v_s);
    float dIdx = SOBEL_SCALE * (*geometry::PointerAt<float>(target_dx_color,
                                                            width, u_t, v_t));
    float dIdy = SOBEL_SCALE * (*geometry::PointerAt<float>(target_dy_color,
                                                            width, u_t, v_t));
    Eigen::Vector3f p3d_mat(
            *geometry::PointerAt<float>(source_xyz, width, 3, u_s, v_s, 0),
            *geometry::PointerAt<float>(source_xyz, width, 3, u_s, v_s, 1),
            *geometry::PointerAt<float>(source_xyz, width, 3, u_s, v_s, 2));
    Eigen::Vector3f p3d_trans = R * p3d_mat + t;
    float invz = 1. / p3d_trans(2);
    float c0 = dIdx * intrinsic(0, 0) * invz;
    float c1 = dIdy * intrinsic(1, 1) * invz;
    float c2 = -(c0 * p3d_trans(0) + c1 * p3d_trans(1)) * invz;

    J_r[0](0) = -p3d_trans(2) * c1 + p3d_trans(1) * c2;
    J_r[0](1) = p3d_trans(2) * c0 - p3d_trans(0) * c2;
    J_r[0](2) = -p3d_trans(1) * c0 + p3d_trans(0) * c1;
    J_r[0](3) = c0;
    J_r[0](4) = c1;
    J_r[0](5) = c2;
    r[0] = diff;
    J_r[1].setZero();
    r[1] = 0.0;
}

__host__ __device__ inline void
RGBDOdometryJacobianFromHybridTerm::ComputeJacobianAndResidual(
        int row,
        Eigen::Vector6f J_r[2],
        float r[2],
        const uint8_t *source_color,
        const uint8_t *source_depth,
        const uint8_t *target_color,
        const uint8_t *target_depth,
        const uint8_t *source_xyz,
        const uint8_t *target_dx_color,
        const uint8_t *target_dx_depth,
        const uint8_t *target_dy_color,
        const uint8_t *target_dy_depth,
        int width,
        const Eigen::Matrix3f &intrinsic,
        const Eigen::Matrix4f &extrinsic,
        const Eigen::Vector4i *corresps) const {
    float sqrt_lamba_dep, sqrt_lambda_img;
    sqrt_lamba_dep = sqrt(LAMBDA_HYBRID_DEPTH);
    sqrt_lambda_img = sqrt(1.0 - LAMBDA_HYBRID_DEPTH);

    const float fx = intrinsic(0, 0);
    const float fy = intrinsic(1, 1);
    Eigen::Matrix3f R = extrinsic.block<3, 3>(0, 0);
    Eigen::Vector3f t = extrinsic.block<3, 1>(0, 3);

    Eigen::Vector4i corresp = corresps[row];
    int u_s = corresp(0);
    int v_s = corresp(1);
    int u_t = corresp(2);
    int v_t = corresp(3);
    float diff_photo =
            (*geometry::PointerAt<float>(target_color, width, u_t, v_t) -
             *geometry::PointerAt<float>(source_color, width, u_s, v_s));
    float dIdx = SOBEL_SCALE * (*geometry::PointerAt<float>(target_dx_color,
                                                            width, u_t, v_t));
    float dIdy = SOBEL_SCALE * (*geometry::PointerAt<float>(target_dy_color,
                                                            width, u_t, v_t));
    float dDdx = SOBEL_SCALE * (*geometry::PointerAt<float>(target_dx_depth,
                                                            width, u_t, v_t));
    float dDdy = SOBEL_SCALE * (*geometry::PointerAt<float>(target_dy_depth,
                                                            width, u_t, v_t));
    if (isnan(dDdx)) dDdx = 0;
    if (isnan(dDdy)) dDdy = 0;
    Eigen::Vector3f p3d_mat(
            *geometry::PointerAt<float>(source_xyz, width, 3, u_s, v_s, 0),
            *geometry::PointerAt<float>(source_xyz, width, 3, u_s, v_s, 1),
            *geometry::PointerAt<float>(source_xyz, width, 3, u_s, v_s, 2));
    Eigen::Vector3f p3d_trans = R * p3d_mat + t;

    float diff_geo =
            *geometry::PointerAt<float>(target_depth, width, u_t, v_t) -
            p3d_trans(2);
    float invz = 1. / p3d_trans(2);
    float c0 = dIdx * fx * invz;
    float c1 = dIdy * fy * invz;
    float c2 = -(c0 * p3d_trans(0) + c1 * p3d_trans(1)) * invz;
    float d0 = dDdx * fx * invz;
    float d1 = dDdy * fy * invz;
    float d2 = -(d0 * p3d_trans(0) + d1 * p3d_trans(1)) * invz;

    J_r[0](0) = sqrt_lambda_img * (-p3d_trans(2) * c1 + p3d_trans(1) * c2);
    J_r[0](1) = sqrt_lambda_img * (p3d_trans(2) * c0 - p3d_trans(0) * c2);
    J_r[0](2) = sqrt_lambda_img * (-p3d_trans(1) * c0 + p3d_trans(0) * c1);
    J_r[0](3) = sqrt_lambda_img * (c0);
    J_r[0](4) = sqrt_lambda_img * (c1);
    J_r[0](5) = sqrt_lambda_img * (c2);
    float r_photo = sqrt_lambda_img * diff_photo;
    r[0] = r_photo;

    J_r[1](0) = sqrt_lamba_dep *
                ((-p3d_trans(2) * d1 + p3d_trans(1) * d2) - p3d_trans(1));
    J_r[1](1) = sqrt_lamba_dep *
                ((p3d_trans(2) * d0 - p3d_trans(0) * d2) + p3d_trans(0));
    J_r[1](2) = sqrt_lamba_dep * ((-p3d_trans(1) * d0 + p3d_trans(0) * d1));
    J_r[1](3) = sqrt_lamba_dep * (d0);
    J_r[1](4) = sqrt_lamba_dep * (d1);
    J_r[1](5) = sqrt_lamba_dep * (d2 - 1.0f);
    float r_geo = sqrt_lamba_dep * diff_geo;
    r[1] = r_geo;
}

}  // namespace odometry
}  // namespace cupoch