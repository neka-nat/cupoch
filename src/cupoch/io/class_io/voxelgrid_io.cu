#include "cupoch/io/class_io/voxelgrid_io.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/utility/helper.h"

using namespace cupoch;
using namespace cupoch::io;


void HostVoxelGrid::FromDevice(const geometry::VoxelGrid& voxelgrid) {
    voxels_keys_.resize(voxelgrid.voxels_keys_.size());
    voxels_values_.resize(voxelgrid.voxels_values_.size());
    utility::CopyFromDeviceMultiStream(voxelgrid.voxels_keys_, voxels_keys_);
    utility::CopyFromDeviceMultiStream(voxelgrid.voxels_values_, voxels_values_);
    cudaDeviceSynchronize();
}

void HostVoxelGrid::ToDevice(geometry::VoxelGrid& voxelgrid) const {
    voxelgrid.voxels_keys_.resize(voxels_keys_.size());
    voxelgrid.voxels_values_.resize(voxels_values_.size());
    utility::CopyToDeviceMultiStream(voxels_keys_, voxelgrid.voxels_keys_);
    utility::CopyToDeviceMultiStream(voxels_values_, voxelgrid.voxels_values_);
    cudaDeviceSynchronize();
}

void HostVoxelGrid::Clear() {
    voxels_keys_.clear();
    voxels_values_.clear();
}
