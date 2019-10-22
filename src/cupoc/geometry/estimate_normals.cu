#include "cupoc/geometry/pointcloud.h"
#include "cupoc/geometry/kdtree_flann.h"

using namespace cupoc;
using namespace cupoc::geometry;

bool PointCloud::EstimateNormals(const KDTreeSearchParam &search_param) {
    const bool has_normal = HasNormals();
    if (HasNormals() == false) {
        normals_.resize(points_.size());
    }
    KDTreeFlann kdtree;
    kdtree.SetGeometry(*this);
    return true;
}