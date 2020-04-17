#include "cupoch/collision/collision.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/geometry/intersection_test.h"

namespace cupoch {
namespace collision {

struct intersect_voxel_voxel_functor {
    intersect_voxel_voxel_functor(const Eigen::Vector3i* voxels_keys1,
                                  const Eigen::Vector3i* voxels_keys2,
                                  float voxel_size1, float voxel_size2,
                                  const Eigen::Vector3f origin1,
                                  const Eigen::Vector3f origin2,
                                  int n_v2)
                                  : voxels_keys1_(voxels_keys1), voxels_keys2_(voxels_keys2),
                                  voxel_size1_(voxel_size1), voxel_size2_(voxel_size2),
                                  box_half_size1_(Eigen::Vector3f(
                                    voxel_size1 / 2, voxel_size1 / 2, voxel_size1 / 2)),
                                  box_half_size2_(Eigen::Vector3f(
                                        voxel_size2 / 2, voxel_size2 / 2, voxel_size2 / 2)),
                                  origin1_(origin1), origin2_(origin2), n_v2_(n_v2) {};
    const Eigen::Vector3i* voxels_keys1_;
    const Eigen::Vector3i* voxels_keys2_;
    const float voxel_size1_;
    const float voxel_size2_;
    const Eigen::Vector3f box_half_size1_;
    const Eigen::Vector3f box_half_size2_;
    const Eigen::Vector3f origin1_;
    const Eigen::Vector3f origin2_;
    const int n_v2_;
    __device__ int operator() (size_t idx) const {
        int i1 = idx / n_v2_;
        int i2 = idx % n_v2_;
        const Eigen::Vector3f h3 = Eigen::Vector3f(0.5, 0.5, 0.5);
        Eigen::Vector3f center1 = ((voxels_keys1_[i1].cast<float>() + h3) * voxel_size1_) + origin1_;
        Eigen::Vector3f center2 = ((voxels_keys2_[i2].cast<float>() + h3) * voxel_size2_) + origin2_;
        int coll = geometry::intersection_test::AABBAABB(center1 - box_half_size1_, center1 + box_half_size1_,
                                                         center2 - box_half_size2_, center2 + box_half_size2_);
        return coll;
    }
};

bool ComputeIntersection(const geometry::VoxelGrid& voxelgrid1,
                         const geometry::VoxelGrid& voxelgrid2) {
    size_t n_v1 = voxelgrid1.voxels_keys_.size();
    size_t n_v2 = voxelgrid2.voxels_keys_.size();
    size_t n_total = n_v1 * n_v2;
    intersect_voxel_voxel_functor func(thrust::raw_pointer_cast(voxelgrid1.voxels_keys_.data()),
                                       thrust::raw_pointer_cast(voxelgrid2.voxels_keys_.data()),
                                       voxelgrid1.voxel_size_, voxelgrid2.voxel_size_,
                                       voxelgrid1.origin_, voxelgrid2.origin_, n_v2);
    int n_coll = thrust::transform_reduce(thrust::make_counting_iterator<size_t>(0),
                                          thrust::make_counting_iterator(n_total),
                                          func, 0, thrust::plus<int>());
    return n_coll > 0;
}

}
}