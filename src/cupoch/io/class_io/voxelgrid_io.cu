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
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/io/class_io/voxelgrid_io.h"
#include "cupoch/utility/helper.h"
#include "cupoch/utility/platform.h"

using namespace cupoch;
using namespace cupoch::io;

void HostVoxelGrid::FromDevice(const geometry::VoxelGrid& voxelgrid) {
    voxels_keys_.resize(voxelgrid.voxels_keys_.size());
    voxels_values_.resize(voxelgrid.voxels_values_.size());
    copy_device_to_host(voxelgrid.voxels_keys_, voxels_keys_);
    copy_device_to_host(voxelgrid.voxels_values_, voxels_values_);
}

void HostVoxelGrid::ToDevice(geometry::VoxelGrid& voxelgrid) const {
    voxelgrid.voxels_keys_.resize(voxels_keys_.size());
    voxelgrid.voxels_values_.resize(voxels_values_.size());
    copy_host_to_device(voxels_keys_, voxelgrid.voxels_keys_);
    copy_host_to_device(voxels_values_, voxelgrid.voxels_values_);
}

void HostVoxelGrid::Clear() {
    voxels_keys_.clear();
    voxels_values_.clear();
}
