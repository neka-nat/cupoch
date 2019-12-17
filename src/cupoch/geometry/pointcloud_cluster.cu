#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct initialize_cluster_matrix_functor {
    initialize_cluster_matrix_functor(const int* indices, const float* dists2,
                                      float eps, int n_points, int* cluster_matrix)
        : indices_(indices), dists2_(dists2), eps_(eps),
          n_points_(n_points), cluster_matrix_(cluster_matrix) {};
    const int* indices_;
    const float* dists2_;
    const float eps_;
    const int n_points_;
    int* cluster_matrix_;
    __device__
    void operator() (size_t idx) {
        cluster_matrix_[idx * n_points_ + idx] = 1;
        for (int i = 0; i < NUM_MAX_NN; ++i) {
            if (indices_[i] < 0) continue;
            if (dists2_[i] <= eps_) {
                cluster_matrix_[indices_[i] * n_points_ + idx] = 1;
            }
        }
    }
};

}

thrust::device_vector<int> PointCloud::ClusterDBSCAN(float eps,
                                                     size_t min_points,
                                                     bool print_progress) const {
    KDTreeFlann kdtree(*this);
    // precompute all neighbours
    utility::LogDebug("Precompute Neighbours");
    thrust::device_vector<int> indices;
    thrust::device_vector<float> dists2;
    kdtree.SearchRadius(points_, eps, indices, dists2);

    const size_t n_pt = points_.size();
    thrust::device_vector<int> cluster_matrix(n_pt * n_pt);
    initialize_cluster_matrix_functor func(thrust::raw_pointer_cast(indices.data()),
                                           thrust::raw_pointer_cast(dists2.data()),
                                           eps, n_pt,
                                           thrust::raw_pointer_cast(cluster_matrix.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(n_pt), func);

    thrust::device_vector<int> labels(points_.size(), -2);
    return labels;
}