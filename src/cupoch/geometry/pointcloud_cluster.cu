#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/utility/console.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct compute_vertex_degree_functor {
    compute_vertex_degree_functor(int *indices, int min_points, int max_edges)
     : indices_(indices), min_points_(min_points), max_edges_(max_edges) {};
    int *indices_;
    const int min_points_;
    const int max_edges_;
    __device__ int operator() (size_t idx) const {
        int count = 0;
        for (int k = 0; k < max_edges_; k++) {
            if (indices_[idx * max_edges_ + k] >= 0) {
                if (indices_[idx * max_edges_ + k] == idx) {
                    indices_[idx * max_edges_ + k] = -1;
                } else {
                    count++;
                }
            }
        }
        if (count >= min_points_) return count;
        for (int k = 0; k < max_edges_; k++) {
            indices_[idx * max_edges_ + k] = -1;
        }
        return 0;
    }
};

struct bfs_functor {
    bfs_functor(const int *vertex_degrees, const int *exscan_vd,
                const int *indices, int* xa, int *fa)
                : vertex_degrees_(vertex_degrees), exscan_vd_(exscan_vd),
                  indices_(indices), xa_(xa), fa_(fa) {};
    const int *vertex_degrees_;
    const int *exscan_vd_;
    const int *indices_;
    int* xa_;
    int* fa_;
    __device__ void operator() (size_t idx) const {
        if (fa_[idx] == 1) {
            fa_[idx] = 0;
            xa_[idx] = 1;
            for (int i = 0; i < vertex_degrees_[idx]; i++) {
                if (xa_[indices_[exscan_vd_[idx] + i]] == 0) {
                    fa_[indices_[exscan_vd_[idx] + i]] = 1;
                }
            }
        }
    }
};

struct set_label_functor {
    set_label_functor(const int *xa, int cluster, int *clusters, int *visited)
     : xa_(xa), cluster_(cluster), clusters_(clusters), visited_(visited) {};
    const int *xa_;
    const int cluster_;
    int *clusters_;
    int *visited_;
    __device__ void operator() (size_t idx) const {
        if (xa_[idx] == 1) {
            clusters_[idx] = cluster_;
            visited_[idx] = 1;
        }
    }
};

}  // namespace

// https://www.sciencedirect.com/science/article/pii/S1877050913003438
utility::device_vector<int> PointCloud::ClusterDBSCAN(
        float eps, size_t min_points, bool print_progress, size_t max_edges) const {
    // precompute all neighbours
    utility::LogDebug("Precompute Neighbours");
    utility::ConsoleProgressBar progress_bar(
            points_.size(), "Precompute Neighbours", print_progress);

    const size_t n_pt = points_.size();
    // Graph construction
    utility::device_vector<int> vertex_degrees(n_pt);
    utility::device_vector<int> exscan_vd(n_pt);
    utility::device_vector<int> indices;
    utility::device_vector<float> distances;
    KDTreeFlann kdtree(*this);
    kdtree.SearchHybrid(points_, eps, max_edges + 1, indices, distances);
    compute_vertex_degree_functor vd_func(thrust::raw_pointer_cast(indices.data()),
                                          min_points, max_edges + 1);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(n_pt), vertex_degrees.begin(), vd_func);
    thrust::exclusive_scan(vertex_degrees.begin(), vertex_degrees.end(), exscan_vd.begin(), 0);
    auto end = thrust::remove_if(indices.begin(), indices.end(),
                                 [] __device__ (int idx) {return idx < 0;});
    indices.resize(thrust::distance(indices.begin(), end));

    // Cluster identification
    int cluster = 0;
    utility::device_vector<int> visited(n_pt, 0);
    thrust::host_vector<int> h_visited(n_pt, 0);
    utility::device_vector<int> clusters(n_pt, -1);
    utility::device_vector<int> xa(n_pt);
    utility::device_vector<int> fa(n_pt);
    for (int i = 0; i < n_pt; i++) {
        ++progress_bar;
        if (h_visited[i] != 1) {
            thrust::fill_n(make_tuple_iterator(visited.begin() + i, clusters.begin() + i),
                           1, thrust::make_tuple(1, cluster));
            thrust::fill(make_tuple_begin(xa, fa), make_tuple_end(xa, fa), thrust::make_tuple(0, 0));
            fa[i] = 1;
            while (thrust::find(fa.begin(), fa.end(), 1) != fa.end()) {
                bfs_functor bfs_func(thrust::raw_pointer_cast(vertex_degrees.data()),
                                     thrust::raw_pointer_cast(exscan_vd.data()),
                                     thrust::raw_pointer_cast(indices.data()),
                                     thrust::raw_pointer_cast(xa.data()),
                                     thrust::raw_pointer_cast(fa.data()));
                thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                                 thrust::make_counting_iterator(n_pt),
                                 bfs_func);
            }
            set_label_functor sl_func(thrust::raw_pointer_cast(xa.data()), cluster,
                                      thrust::raw_pointer_cast(clusters.data()),
                                      thrust::raw_pointer_cast(visited.data()));
            thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                             thrust::make_counting_iterator(n_pt), sl_func);
            h_visited = visited;
            cluster++;
        }
    }
    return clusters;
}