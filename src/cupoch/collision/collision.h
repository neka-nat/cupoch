#pragma once

namespace cupoch {

namespace geometry {
class VoxelGrid;
class OccupancyGrid;
}

namespace collision {

bool ComputeIntersection(const geometry::VoxelGrid& voxelgrid1,
                         const geometry::VoxelGrid& voxelgrid2);

bool ComputeIntersection(const geometry::VoxelGrid& voxelgrid,
                         const geometry::OccupancyGrid& occgrid);

bool ComputeIntersection(const geometry::OccupancyGrid& occgrid,
                         const geometry::VoxelGrid& voxelgrid);

}
}