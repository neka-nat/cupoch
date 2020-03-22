#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct initialize_cluster_matrix_functor {
    initialize_cluster_matrix_functor(const Eigen::Vector3f* points,
                                      float eps,
                                      int n_points,
                                      char *cluster_matrix)
        : points_(points),
          eps_(eps),
          n_points_(n_points),
          cluster_matrix_(cluster_matrix) {};
    const Eigen::Vector3f* points_;
    const float eps_;
    const int n_points_;
    char *cluster_matrix_;
    __device__ void operator()(size_t idx) {
        // cluster_matrix
        //            1st class | 2nd class | ... | N-th class
        // 1st point  [1          0           ...  0          ]
        // 2nd point  [0          1           ...  0          ]
        // ...         ...
        // N-th point [0          0           ...  1          ]
        int k = idx / n_points_;
        int i = idx % n_points_;
        if (i == k) {
            cluster_matrix_[idx] = 1;
        } else if ((points_[k] - points_[i]).norm() < eps_) {
            cluster_matrix_[idx] = 1;
        }
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
        // follow the route until a valid class is found
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
        if (target_cluster < 0 || idx == target_cluster) return 0;
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
                // make a point in the idx class belong to the target class
                cluster_matrix_[i * n_points_ + target_cluster] |= cluster_matrix_[i * n_points_ + idx];
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
        if (!valid_[idx]) return; // check idx class is valid
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
    // precompute all neighbours
    utility::LogDebug("Precompute Neighbours");
    utility::ConsoleProgressBar progress_bar(
            points_.size(), "Precompute Neighbours", print_progress);

    const size_t n_pt = points_.size();
    utility::device_vector<char> cluster_matrix(n_pt * n_pt, 0);
    utility::device_vector<char> valid(n_pt, 1);
    utility::device_vector<int> reroute(n_pt, -1);
    initialize_cluster_matrix_functor func(
            thrust::raw_pointer_cast(points_.data()),
            eps, n_pt,
            thrust::raw_pointer_cast(cluster_matrix.data()));
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(n_pt * n_pt), func);

    for (int i = 0; i < n_pt; ++i) { // cluster loop
        ++progress_bar;
        bool is_merged = true;
        merge_cluster_functor func(
                i, thrust::raw_pointer_cast(cluster_matrix.data()),
                thrust::raw_pointer_cast(valid.data()),
                thrust::raw_pointer_cast(reroute.data()), n_pt);
        while (is_merged) {
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