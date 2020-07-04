#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/io/class_io/voxelgrid_io.h"
#include "cupoch/utility/helper.h"

using namespace cupoch;
using namespace cupoch::io;

void HostVoxelGrid::FromDevice(const geometry::VoxelGrid& voxelgrid) {
    voxels_keys_.resize(voxelgrid.voxels_keys_.size());
    voxels_values_.resize(voxelgrid.voxels_values_.size());
    thrust::copy(voxelgrid.voxels_keys_.begin(), voxelgrid.voxels_keys_.end(),
                 voxels_keys_.begin());
    thrust::copy(voxelgrid.voxels_values_.begin(),
                 voxelgrid.voxels_values_.end(), voxels_values_.begin());
}

void HostVoxelGrid::ToDevice(geometry::VoxelGrid& voxelgrid) const {
    voxelgrid.voxels_keys_.resize(voxels_keys_.size());
    voxelgrid.voxels_values_.resize(voxels_values_.size());
    thrust::copy(voxels_keys_.begin(), voxels_keys_.end(),
                 voxelgrid.voxels_keys_.begin());
    thrust::copy(voxels_values_.begin(), voxels_values_.end(),
                 voxelgrid.voxels_values_.begin());
}

void HostVoxelGrid::Clear() {
    voxels_keys_.clear();
    voxels_values_.clear();
}
