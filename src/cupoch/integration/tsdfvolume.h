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

#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/rgbdimage.h"
#include "cupoch/geometry/trianglemesh.h"

namespace cupoch {
namespace integration {

enum class TSDFVolumeColorType {
    NoColor = 0,
    RGB8 = 1,
    Gray32 = 2,
};

/// Interface class of the Truncated Signed Distance Function (TSDF) volume
/// This volume is usually used to integrate surface data (e.g., a series of
/// RGB-D images) into a Mesh or PointCloud. The basic technique is presented in
/// the following paper:
/// B. Curless and M. Levoy
/// A volumetric method for building complex models from range images
/// In SIGGRAPH, 1996
class TSDFVolume {
public:
    TSDFVolume(float voxel_length,
               float sdf_trunc,
               TSDFVolumeColorType color_type)
        : voxel_length_(voxel_length),
          sdf_trunc_(sdf_trunc),
          color_type_(color_type) {}
    virtual ~TSDFVolume() {}

public:
    /// Function to reset the TSDFVolume
    virtual void Reset() = 0;

    /// Function to integrate an RGB-D image into the volume
    virtual void Integrate(const geometry::RGBDImage &image,
                           const camera::PinholeCameraIntrinsic &intrinsic,
                           const Eigen::Matrix4f &extrinsic) = 0;

    /// Function to extract a point cloud with normals
    virtual std::shared_ptr<geometry::PointCloud> ExtractPointCloud() = 0;

    /// Function to extract a triangle mesh, using the marching cubes algorithm
    /// (https://en.wikipedia.org/wiki/Marching_cubes)
    virtual std::shared_ptr<geometry::TriangleMesh> ExtractTriangleMesh() = 0;

public:
    float voxel_length_;
    float sdf_trunc_;
    TSDFVolumeColorType color_type_;
};

}  // namespace integration
}  // namespace cupoch