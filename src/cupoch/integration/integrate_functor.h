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
#pragma once
#include "cupoch/integration/uniform_tsdfvolume.h"

namespace cupoch {
namespace integration {

struct integrate_functor {
    integrate_functor(float fx,
                      float fy,
                      float cx,
                      float cy,
                      const Eigen::Matrix4f &extrinsic,
                      float voxel_length,
                      float sdf_trunc,
                      float safe_width,
                      float safe_height,
                      int resolution,
                      const uint8_t *color,
                      const uint8_t *depth,
                      const uint8_t *depth_to_camera_distance_multiplier,
                      int width,
                      int num_of_channels,
                      TSDFVolumeColorType color_type)
        : fx_(fx),
          fy_(fy),
          cx_(cx),
          cy_(cy),
          extrinsic_(extrinsic),
          voxel_length_(voxel_length),
          half_voxel_length_(0.5 * voxel_length),
          sdf_trunc_(sdf_trunc),
          sdf_trunc_inv_(1.0 / sdf_trunc),
          extrinsic_scaled_(voxel_length * extrinsic),
          safe_width_(safe_width),
          safe_height_(safe_height),
          resolution_(resolution),
          color_(color),
          depth_(depth),
          depth_to_camera_distance_multiplier_(
                  depth_to_camera_distance_multiplier),
          width_(width),
          num_of_channels_(num_of_channels),
          color_type_(color_type) {}
    const float fx_;
    const float fy_;
    const float cx_;
    const float cy_;
    const Eigen::Matrix4f extrinsic_;
    const float voxel_length_;
    const float half_voxel_length_;
    const float sdf_trunc_;
    const float sdf_trunc_inv_;
    const Eigen::Matrix4f extrinsic_scaled_;
    const float safe_width_;
    const float safe_height_;
    const int resolution_;
    const uint8_t *color_;
    const uint8_t *depth_;
    const uint8_t *depth_to_camera_distance_multiplier_;
    const int width_;
    const int num_of_channels_;
    const TSDFVolumeColorType color_type_;
    __device__ virtual ~integrate_functor() {};
    __device__ void ComputeTSDF(geometry::TSDFVoxel &voxel, const Eigen::Vector3f& origin,
                                int x, int y, int z) {
        Eigen::Vector4f pt_3d_homo(
                float(half_voxel_length_ + voxel_length_ * x + origin(0)),
                float(half_voxel_length_ + voxel_length_ * y + origin(1)),
                float(half_voxel_length_ + origin(2)), 1.f);
        Eigen::Vector4f pt_camera = extrinsic_ * pt_3d_homo;
        pt_camera(0) += z * extrinsic_scaled_(0, 2);
        pt_camera(1) += z * extrinsic_scaled_(1, 2);
        pt_camera(2) += z * extrinsic_scaled_(2, 2);
        // Skip if negative depth after projection
        if (pt_camera(2) <= 0) {
            return;
        }
        // Skip if x-y coordinate not in range
        float u_f = pt_camera(0) * fx_ / pt_camera(2) + cx_ + 0.5f;
        float v_f = pt_camera(1) * fy_ / pt_camera(2) + cy_ + 0.5f;
        if (!(u_f >= 0.0001f && u_f < safe_width_ && v_f >= 0.0001f &&
              v_f < safe_height_)) {
            return;
        }
        // Skip if negative depth in depth image
        int u = (int)u_f;
        int v = (int)v_f;
        float d = *geometry::PointerAt<float>(depth_, width_, u, v);
        if (d <= 0.0f) {
            return;
        }

        float sdf =
                (d - pt_camera(2)) *
                (*geometry::PointerAt<float>(
                        depth_to_camera_distance_multiplier_, width_, u, v));
        if (sdf > -sdf_trunc_) {
            // integrate
            float tsdf = min(1.0f, sdf * sdf_trunc_inv_);
            const geometry::TSDFVoxel cv = voxel;
            voxel.tsdf_ = (cv.tsdf_ * cv.weight_ + tsdf) /
                          (cv.weight_ + 1.0f);
            if (color_type_ == TSDFVolumeColorType::RGB8) {
                const uint8_t *rgb = geometry::PointerAt<uint8_t>(
                        color_, width_, num_of_channels_, u, v, 0);
                Eigen::Vector3f rgb_f(rgb[0], rgb[1], rgb[2]);
                voxel.color_ = (cv.color_ * cv.weight_ + rgb_f) /
                               (cv.weight_ + 1.0f);
            } else if (color_type_ == TSDFVolumeColorType::Gray32) {
                const float intensity = *geometry::PointerAt<float>(
                        color_, width_, num_of_channels_, u, v, 0);
                voxel.color_ =
                        (cv.color_.array() * cv.weight_ + intensity) /
                        (cv.weight_ + 1.0f);
            }
            voxel.weight_ += 1.0f;
        }
    }
};

struct uniform_integrate_functor : public integrate_functor {
    uniform_integrate_functor(float fx,
                              float fy,
                              float cx,
                              float cy,
                              const Eigen::Matrix4f &extrinsic,
                              float voxel_length,
                              float sdf_trunc,
                              float safe_width,
                              float safe_height,
                              int resolution,
                              const uint8_t *color,
                              const uint8_t *depth,
                              const uint8_t *depth_to_camera_distance_multiplier,
                              int width,
                              int num_of_channels,
                              TSDFVolumeColorType color_type,
                              const Eigen::Vector3f &origin,
                              geometry::TSDFVoxel *voxels)
        : integrate_functor(fx, fy, cx, cy,
                            extrinsic, voxel_length,
                            sdf_trunc, safe_width,
                            safe_height, resolution,
                            color, depth,
                            depth_to_camera_distance_multiplier,
                            width, num_of_channels,
                            color_type),
          origin_(origin), voxels_(voxels) {};
    const Eigen::Vector3f origin_;
    geometry::TSDFVoxel *voxels_;
    __device__ void operator() (size_t idx) {
        int res2 = resolution_ * resolution_;
        int x = idx / res2;
        int yz = idx % res2;
        int y = yz / resolution_;
        int z = yz % resolution_;
        ComputeTSDF(voxels_[idx], origin_, x, y, z);
    }
};

}
}