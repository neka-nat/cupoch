#pragma once

namespace cupoch {

namespace geometry {
class VoxelGrid;
}

namespace collision {

bool ComputeIntersection(const geometry::VoxelGrid& voxelgrid1,
                         const geometry::VoxelGrid& voxelgrid2);

}
}