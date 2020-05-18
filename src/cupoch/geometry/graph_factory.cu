#include "cupoch/geometry/graph.h"
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/trianglemesh.h"

namespace cupoch {
namespace geometry {

namespace {

__constant__ int voxel_offset[26][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0},
                                        {0, 0, 1}, {0, 0, -1}, {1, 1, 0}, {1, -1, 0},
                                        {-1, 1, 0}, {-1, -1, 0}, {1, 0, 1}, {1, 0, -1},
                                        {-1, 0, 1}, {-1, 0, -1}, {0, 1, 1}, {0, 1, -1},
                                        {0, -1, 1}, {0, -1, -1}, {1, 1, 1}, {-1, -1, -1},
                                        {-1, 1, 1}, {1, -1, -1}, {1, -1, 1}, {-1, 1, -1},
                                        {-1, -1, 1}, {1, 1, -1}};

struct create_dense_grid_points_functor {
    create_dense_grid_points_functor(const Eigen::Vector3i& resolutions,
                                     const Eigen::Vector3f& min_bound,
                                     const Eigen::Vector3f& steps)
                                     : resolutions_(resolutions), min_bound_(min_bound),
                                     steps_(steps) {};
    const Eigen::Vector3i resolutions_;
    const Eigen::Vector3f min_bound_;
    const Eigen::Vector3f steps_;
    __device__ Eigen::Vector3f operator() (size_t idx) {
        int x = idx / (resolutions_[1] * resolutions_[2]);
        int yz = idx % (resolutions_[1] * resolutions_[2]);
        int y = yz / resolutions_[2];
        int z = yz % resolutions_[2];
        return min_bound_ + (Eigen::Vector3i(x, y, z).cast<float>().array() * steps_.array()).matrix();
    }
};

struct create_dense_grid_lines_functor {
    create_dense_grid_lines_functor(const Eigen::Vector3i& resolutions) : resolutions_(resolutions) {};
    const Eigen::Vector3i resolutions_;
    __device__ Eigen::Vector2i operator() (size_t idx) const {
        int x = idx / (resolutions_[1] * resolutions_[2] * 26);
        int yzk = idx % (resolutions_[1] * resolutions_[2] * 26);
        int y = yzk / (resolutions_[2] * 26);
        int zk = yzk % (resolutions_[2] * 26);
        int z = zk / 26;
        int k = zk % 26;
        Eigen::Vector3i gidx = Eigen::Vector3i(x + voxel_offset[k][0],
                                               y + voxel_offset[k][1],
                                               z + voxel_offset[k][2]);
        int j = gidx[0] * resolutions_[1] * resolutions_[2]  + gidx[1] * resolutions_[2] + gidx[2];
        if (j < 0 || j >= resolutions_.prod()) return Eigen::Vector2i(-1, -1);
        return Eigen::Vector2i(idx, j);
    }
};

}

std::shared_ptr<Graph> Graph::CreateFromTriangleMesh(const TriangleMesh &input) {
    auto out = std::make_shared<Graph>();
    out->points_ = input.vertices_;
    if (input.HasEdgeList()) {
        out->lines_ = input.edge_list_;
    } else {
        TriangleMesh tmp;
        tmp.triangles_ = input.triangles_;
        tmp.ComputeEdgeList();
        out->lines_ = tmp.edge_list_;
    }
    out->ConstructGraph();
    return out;
}

std::shared_ptr<Graph> Graph::CreateFromAxisAlignedBoundingBox(const geometry::AxisAlignedBoundingBox& bbox,
                                                               const Eigen::Vector3i& resolutions) {
    auto out = std::make_shared<Graph>();
    Eigen::Vector3f steps = (bbox.max_bound_ - bbox.min_bound_).array() / resolutions.cast<float>().array();
    size_t n_points = resolutions.prod();
    out->points_.resize(n_points);
    create_dense_grid_points_functor pfunc(resolutions, bbox.min_bound_, steps);
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(n_points),
                      out->points_.begin(), pfunc);
    out->lines_.resize(n_points * 26);
    create_dense_grid_lines_functor lfunc(resolutions);
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(n_points),
                      out->lines_.begin(), lfunc);
    auto end = thrust::remove_if(out->lines_.begin(), out->lines_.end(),
                                 [] __device__ (const Eigen::Vector2i& l) {
                                     return l[0] < 0;
                                 });
    out->lines_.resize(thrust::distance(out->lines_.begin(), end));
    out->ConstructGraph();
    return out;
}

}
}