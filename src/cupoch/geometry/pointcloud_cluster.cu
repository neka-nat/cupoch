#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct initialize_cluster_matrix_functor {
    initialize_cluster_matrix_functor(const int *indices,
                                      const float *dists2,
                                      float eps,
                                      int n_points,
                                      char *cluster_matrix,
                                      char *valid,
                                      int *reroute)
        : indices_(indices),
          dists2_(dists2),
          eps_(eps),
          n_points_(n_points),
          cluster_matrix_(cluster_matrix),
          valid_(valid),
          reroute_(reroute){};
    const int *indices_;
    const float *dists2_;
    const float eps_;
    const int n_points_;
    char *cluster_matrix_;
    char *valid_;
    int *reroute_;
    __device__ void operator()(size_t idx) {
        cluster_matrix_[idx * n_points_ + idx] = 1;
        for (int k = 0; k < NUM_MAX_NN; ++k) {
            if (indices_[idx * NUM_MAX_NN + k] < 0) continue;
            if (dists2_[idx * NUM_MAX_NN + k] <= eps_) {
                cluster_matrix_[indices_[idx * NUM_MAX_NN + k] * n_points_ +
                                idx] = 1;
            }
        }
        valid_[idx] = 1;
        reroute_[idx] = -1;
    }
};

struct merge_cluster_functor {
    merge_cluster_functor(int cluster_index,
                          char *cluster_matrix,
                          char *valid,
                          int *reroute,
                          int n_points)
        : cluster_index_(cluster_index),
          cluster_matrix_(cluster_matrix),
          valid_(valid),
          reroute_(reroute),
          n_points_(n_points){};
    const int cluster_index_;
    char *cluster_matrix_;
    char *valid_;
    int *reroute_;
    const int n_points_;
    __device__ int get_reroute_index(int idx) {
        int ans_idx = idx;
        while (ans_idx != -1) {
            if (valid_[ans_idx]) return ans_idx;
            ans_idx = reroute_[ans_idx];
        }
        return -1;
    }

    __device__ int operator()(int idx) {
        if (valid_[idx] != 1 || idx == cluster_index_) return 0;
        int target_cluster = get_reroute_index(cluster_index_);
        if (idx == target_cluster) return 0;
        bool do_merge = false;
        for (int i = 0; i < n_points_; ++i) {
            if (cluster_matrix_[i * n_points_ + idx] &&
                cluster_matrix_[i * n_points_ + target_cluster]) {
                do_merge = true;
                break;
            }
        }

        if (do_merge) {
            valid_[idx] = 0;
            reroute_[idx] = target_cluster;
            for (int i = 0; i < n_points_; ++i) {
                if (cluster_matrix_[i * n_points_ + idx] == 1) {
                    cluster_matrix_[i * n_points_ + target_cluster] = 1;
                }
            }
            return 1;
        }
        return 0;
    }
};

struct assign_cluster_functor {
    assign_cluster_functor(const char *cluster_matrix,
                           const char *valid,
                           int n_points,
                           int min_points,
                           int *labels)
        : cluster_matrix_(cluster_matrix),
          valid_(valid),
          n_points_(n_points),
          min_points_(min_points),
          labels_(labels){};
    const char *cluster_matrix_;
    const char *valid_;
    const int n_points_;
    const int min_points_;
    int *labels_;
    __device__ void operator()(size_t idx) {
        if (!valid_[idx]) return;
        int count = 0;
        for (int i = 0; i < n_points_; ++i) {
            count += cluster_matrix_[i * n_points_ + idx];
        }
        if (count > min_points_) {
            for (int i = 0; i < n_points_; ++i) {
                if (cluster_matrix_[i * n_points_ + idx] == 1) {
                    labels_[i] = idx;
                }
            }
        }
    }
};

}  // namespace

// https://arxiv.org/pdf/1506.02226.pdf
// https://github.com/Maghoumi/cudbscan
utility::device_vector<int> PointCloud::ClusterDBSCAN(
        float eps, size_t min_points, bool print_progress) const {
    KDTreeFlann kdtree(*this);
    // precompute all neighbours
    utility::LogDebug("Precompute Neighbours");
    utility::ConsoleProgressBar progress_bar(
            points_.size(), "Precompute Neighbours", print_progress);
    utility::device_vector<int> indices;
    utility::device_vector<float> dists2;
    kdtree.SearchRadius(points_, eps, indices, dists2);

    const size_t n_pt = points_.size();
    utility::device_vector<char> cluster_matrix(n_pt * n_pt);
    utility::device_vector<char> valid(n_pt);
    utility::device_vector<int> reroute(n_pt);
    initialize_cluster_matrix_functor func(
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(dists2.data()), eps, n_pt,
            thrust::raw_pointer_cast(cluster_matrix.data()),
            thrust::raw_pointer_cast(valid.data()),
            thrust::raw_pointer_cast(reroute.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(n_pt), func);

    bool is_merged = true;
    for (int i = 0; i < n_pt; ++i) {
        ++progress_bar;
        while (is_merged) {
            merge_cluster_functor func(
                    i, thrust::raw_pointer_cast(cluster_matrix.data()),
                    thrust::raw_pointer_cast(valid.data()),
                    thrust::raw_pointer_cast(reroute.data()), n_pt);
            const int n_merged = thrust::transform_reduce(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator((int)n_pt), func, 0,
                    thrust::plus<int>());
            is_merged = n_merged > 0;
        }
    }

    utility::device_vector<int> labels(points_.size(), -1);
    assign_cluster_functor assign_func(
            thrust::raw_pointer_cast(cluster_matrix.data()),
            thrust::raw_pointer_cast(valid.data()), n_pt, min_points,
            thrust::raw_pointer_cast(labels.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(n_pt), assign_func);
    return labels;
}