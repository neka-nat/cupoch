/**
 * Copyright (c) 2020 Neka-Nat
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/
#include <thrust/sort.h>
#include <thrust/set_operations.h>
#include <thrust/gather.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <stdgpu/unordered_set.cuh>

#include "cupoch/geometry/geometry_functor.h"
#include "cupoch/geometry/graph.h"
#include "cupoch/knn/kdtree_flann.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace geometry {

namespace {

template <int Dim>
struct extract_near_edges_functor {
    extract_near_edges_functor(const Eigen::Matrix<float, Dim, 1> &point,
                               int point_no,
                               float max_edge_distance)
        : point_(point),
          point_no_(point_no),
          max_edge_distance_(max_edge_distance){};
    const Eigen::Matrix<float, Dim, 1> point_;
    const int point_no_;
    const float max_edge_distance_;
    __device__ thrust::tuple<Eigen::Vector2i, float> operator()(
            const thrust::tuple<int, Eigen::Matrix<float, Dim, 1>> &x) const {
        int i = thrust::get<0>(x);
        const Eigen::Matrix<float, Dim, 1> &p = thrust::get<1>(x);
        float d = (p - point_).norm();
        return thrust::make_tuple((d < max_edge_distance_)
                                          ? Eigen::Vector2i(i, point_no_)
                                          : Eigen::Vector2i(-1, -1),
                                  d);
    }
};

template <int Dim>
struct relax_functor {
    relax_functor(const Eigen::Vector2i *lines,
                  const int *edge_index_offsets,
                  const float *edge_weights,
                  const int *edge_table,
                  int *open_flags,
                  const typename Graph<Dim>::SSSPResult *res,
                  typename Graph<Dim>::SSSPResult *res_tmp)
        : lines_(lines),
          edge_index_offsets_(edge_index_offsets),
          edge_weights_(edge_weights),
          edge_table_(edge_table),
          open_flags_(open_flags),
          res_(res),
          res_tmp_(res_tmp){};
    const Eigen::Vector2i *lines_;
    const int *edge_index_offsets_;
    const float *edge_weights_;
    const int *edge_table_;
    int *open_flags_;
    const typename Graph<Dim>::SSSPResult *res_;
    typename Graph<Dim>::SSSPResult *res_tmp_;
    __device__ void operator()(size_t idx) {
        if (open_flags_[idx] == 0) return;
        open_flags_[idx] = 0;
        int s_edge = edge_index_offsets_[idx];
        int e_edge = edge_index_offsets_[idx + 1];
        for (int j = s_edge; j < e_edge; ++j) {
            int k = lines_[j][0];
            res_tmp_[edge_table_[j]].shortest_distance_ =
                    res_[k].shortest_distance_ + edge_weights_[j];
            res_tmp_[edge_table_[j]].prev_index_ = k;
        }
    }
};

template <int Dim>
struct update_shortest_distances_functor {
    update_shortest_distances_functor(
            int *open_flags,
            typename Graph<Dim>::SSSPResult *res,
            const typename Graph<Dim>::SSSPResult *res_tmp)
        : open_flags_(open_flags), res_(res), res_tmp_(res_tmp){};
    int *open_flags_;
    typename Graph<Dim>::SSSPResult *res_;
    const typename Graph<Dim>::SSSPResult *res_tmp_;
    __device__ void operator()(size_t idx) {
        if (res_[idx].shortest_distance_ > res_tmp_[idx].shortest_distance_) {
            res_[idx] = res_tmp_[idx];
            open_flags_[idx] = 1;
        }
    }
};

template <int Dim>
struct compare_path_length_functor {
    compare_path_length_functor(const typename Graph<Dim>::SSSPResult *res,
                                const int *open_flags,
                                int end_node_index)
        : res_(res), open_flags_(open_flags), end_node_index_(end_node_index) {}
    const typename Graph<Dim>::SSSPResult *res_;
    const int *open_flags_;
    const int end_node_index_;
    __device__ bool operator()(size_t idx) const {
        return (open_flags_[idx] &&
                res_[idx].shortest_distance_ <
                        res_[end_node_index_].shortest_distance_);
    }
};

template <class... Args>
struct check_edge_functor {
    check_edge_functor(const Eigen::Vector2i &edge, bool is_directed)
        : edge_(edge), is_directed_(is_directed){};
    const Eigen::Vector2i edge_;
    const bool is_directed_;
    __device__ bool operator()(const thrust::tuple<Args...> &x) const {
        const Eigen::Vector2i &l = thrust::get<0>(x);
        return l == edge_ ||
               (!is_directed_ && l == Eigen::Vector2i(edge_[1], edge_[0]));
    }
};

}  // namespace

template <int Dim>
Graph<Dim>::Graph() : LineSet<Dim>(Geometry::GeometryType::Graph) {}

template <int Dim>
Graph<Dim>::Graph(
        const utility::device_vector<Eigen::Matrix<float, Dim, 1>> &points)
    : LineSet<Dim>(Geometry::GeometryType::Graph,
                   points,
                   utility::device_vector<Eigen::Vector2i>()) {}

template <int Dim>
Graph<Dim>::Graph(
        const thrust::host_vector<Eigen::Matrix<float, Dim, 1>> &points)
    : LineSet<Dim>(Geometry::GeometryType::Graph,
                   points,
                   utility::device_vector<Eigen::Vector2i>()) {
    ConstructGraph();
}

template <int Dim>
Graph<Dim>::~Graph() {}

template <int Dim>
Graph<Dim>::Graph(const Graph &other)
    : LineSet<Dim>(Geometry::GeometryType::Graph, other.points_, other.lines_),
      edge_index_offsets_(other.edge_index_offsets_),
      edge_weights_(other.edge_weights_),
      is_directed_(other.is_directed_) {}

template <int Dim>
std::vector<int> Graph<Dim>::GetEdgeIndexOffsets() const {
    std::vector<int> edge_index_offsets(edge_index_offsets_.size());
    copy_device_to_host(edge_index_offsets_, edge_index_offsets);
    return edge_index_offsets;
}

template <int Dim>
void Graph<Dim>::SetEdgeIndexOffsets(
        const thrust::host_vector<int> &edge_index_offsets) {
    edge_index_offsets_ = edge_index_offsets;
}

template <int Dim>
std::vector<float> Graph<Dim>::GetEdgeWeights() const {
    std::vector<float> edge_weights(edge_weights_.size());
    copy_device_to_host(edge_weights_, edge_weights);
    return edge_weights;
}

template <int Dim>
void Graph<Dim>::SetEdgeWeights(
        const thrust::host_vector<float> &edge_weights) {
    edge_weights_ = edge_weights;
}

template <int Dim>
Graph<Dim> &Graph<Dim>::Clear() {
    LineSet<Dim>::Clear();
    edge_index_offsets_.clear();
    edge_weights_.clear();
    return *this;
}

template <int Dim>
Graph<Dim> &Graph<Dim>::ConstructGraph(bool set_edge_weights_from_distance) {
    if (this->lines_.empty()) {
        utility::LogError("[ConstructGraph] Graph has no edges.");
        return *this;
    }

    bool has_colors = this->HasColors();
    bool has_weights = this->HasWeights();
    if (has_colors && has_weights) {
        thrust::sort_by_key(utility::exec_policy(0),
                            this->lines_.begin(), this->lines_.end(),
                            make_tuple_begin(edge_weights_, this->colors_));
    } else if (!has_colors && has_weights) {
        thrust::sort_by_key(utility::exec_policy(0),
                            this->lines_.begin(), this->lines_.end(),
                            edge_weights_.begin());
    } else if (has_colors && !has_weights) {
        thrust::sort_by_key(utility::exec_policy(0),
                            this->lines_.begin(), this->lines_.end(),
                            this->colors_.begin());
    } else {
        thrust::sort(utility::exec_policy(0), this->lines_.begin(),
                     this->lines_.end());
        edge_weights_.resize(this->lines_.size(), 1.0);
    }
    edge_index_offsets_.resize(this->points_.size() + 1);
    thrust::fill(edge_index_offsets_.begin(), edge_index_offsets_.end(), 0);
    utility::device_vector<int> indices(this->lines_.size());
    utility::device_vector<int> counts(this->lines_.size());
    const auto begin = thrust::make_transform_iterator(
            this->lines_.begin(), element_get_functor<Eigen::Vector2i, 0>());
    auto end = thrust::reduce_by_key(utility::exec_policy(0), begin,
                                     thrust::make_transform_iterator(
                                            this->lines_.end(),
                                            element_get_functor<Eigen::Vector2i, 0>()),
                                     thrust::make_constant_iterator<int>(1),
                                     indices.begin(), counts.begin());
    indices.resize(thrust::distance(indices.begin(), end.first));
    counts.resize(thrust::distance(counts.begin(), end.second));
    thrust::scatter(counts.begin(), counts.end(), indices.begin(),
                    edge_index_offsets_.begin());
    thrust::exclusive_scan(
            utility::exec_policy(0), edge_index_offsets_.begin(),
            edge_index_offsets_.end(), edge_index_offsets_.begin());
    if (set_edge_weights_from_distance) {
        SetEdgeWeightsFromDistance();
    }
    return *this;
}

template <int Dim>
Graph<Dim> &Graph<Dim>::ConnectToNearestNeighbors(float max_edge_distance,
                                                  size_t max_num_edges) {
    utility::device_vector<int> indices;
    utility::device_vector<float> weights;
    utility::device_vector<Eigen::Vector2i> new_edges(this->points_.size() *
                                                      (max_num_edges + 1));
    knn::KDTreeFlann kdtree;
    kdtree.SetRawData(this->points_);
    kdtree.SearchRadius(this->points_, max_edge_distance, max_num_edges + 1,
                        indices, weights);
    thrust::transform(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator<int>(new_edges.size()),
                      indices.begin(), new_edges.begin(),
                      [max_num_edges] __device__(int idx, int j) {
                          int i = idx / max_num_edges;
                          return (j >= 0 && i != j) ? Eigen::Vector2i(i, j)
                                                    : Eigen::Vector2i(-1, -1);
                      });
    auto remove_fn =
            [] __device__(const thrust::tuple<Eigen::Vector2i, float> &x) {
                return thrust::get<0>(x)[0] < 0;
            };
    remove_if_vectors(utility::exec_policy(0), remove_fn, new_edges,
                      weights);
    thrust::sort_by_key(utility::exec_policy(0), new_edges.begin(),
                        new_edges.end(), weights.begin());
    utility::device_vector<Eigen::Vector2i> res_edges(new_edges.size());
    utility::device_vector<float> res_weights(new_edges.size());
    auto end = thrust::set_difference_by_key(new_edges.begin(), new_edges.end(),
                                             this->lines_.begin(), this->lines_.end(),
                                             weights.begin(), edge_weights_.begin(),
                                             res_edges.begin(), res_weights.begin());
    resize_all(thrust::distance(res_edges.begin(), end.first), res_edges, res_weights);
    this->lines_.insert(this->lines_.end(), res_edges.begin(), res_edges.end());
    edge_weights_.insert(edge_weights_.end(), res_weights.begin(),
                         res_weights.end());
    return ConstructGraph(false);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::AddNodeAndConnect(
        const Eigen::Matrix<float, Dim, 1> &point,
        float max_edge_distance,
        bool lazy_add) {
    size_t n_points = this->points_.size();
    utility::device_vector<Eigen::Vector2i> new_edges(n_points);
    utility::device_vector<float> new_weights(n_points);
    extract_near_edges_functor<Dim> func(point, n_points, max_edge_distance);
    thrust::transform(enumerate_begin(this->points_),
                      enumerate_end(this->points_),
                      make_tuple_begin(new_edges, new_weights), func);
    auto remove_fn =
            [] __device__(const thrust::tuple<Eigen::Vector2i, float> &x) {
                return thrust::get<0>(x)[0] < 0;
            };
    remove_if_vectors(utility::exec_policy(0), remove_fn, new_edges,
                      new_weights);
    this->points_.push_back(point);
    return AddEdges(new_edges, new_weights, lazy_add);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::AddEdge(const Eigen::Vector2i &edge,
                                float weight,
                                bool lazy_add) {
    this->lines_.push_back(edge);
    edge_weights_.push_back(weight);
    if (!is_directed_) {
        this->lines_.push_back(Eigen::Vector2i(edge[1], edge[0]));
        edge_weights_.push_back(weight);
    }
    if (this->HasColors()) {
        this->colors_.push_back(Eigen::Vector3f::Ones());
        if (!is_directed_) this->colors_.push_back(Eigen::Vector3f::Ones());
    }
    return (lazy_add) ? *this : ConstructGraph(false);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::AddEdges(
        const utility::device_vector<Eigen::Vector2i> &edges,
        const utility::device_vector<float> &weights,
        bool lazy_add) {
    if (!weights.empty() && edges.size() != weights.size()) {
        utility::LogError(
                "[AddEdges] edges size is not equal to weights size.");
        return *this;
    }
    size_t n_old_lines = this->lines_.size();
    this->lines_.insert(this->lines_.end(), edges.begin(), edges.end());
    if (!is_directed_) {
        this->lines_.insert(this->lines_.end(),
                            thrust::make_transform_iterator(
                                    edges.begin(), swap_index_functor<int>()),
                            thrust::make_transform_iterator(
                                    edges.end(), swap_index_functor<int>()));
    }
    if (weights.empty()) {
        if (!is_directed_) {
            edge_weights_.resize(2 * this->lines_.size());
        } else {
            edge_weights_.resize(this->lines_.size());
        }
        thrust::fill(edge_weights_.begin() + n_old_lines, edge_weights_.end(),
                     1.0);
    } else {
        edge_weights_.insert(edge_weights_.end(), weights.begin(),
                             weights.end());
        if (!is_directed_)
            edge_weights_.insert(edge_weights_.end(), weights.begin(),
                                 weights.end());
    }
    if (this->HasColors()) {
        this->colors_.resize(this->lines_.size());
        thrust::fill(this->colors_.begin() + n_old_lines, this->colors_.end(),
                     Eigen::Vector3f::Ones());
    }
    return (lazy_add) ? *this : ConstructGraph(false);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::AddEdges(
        const utility::pinned_host_vector<Eigen::Vector2i> &edges,
        const utility::pinned_host_vector<float> &weights,
        bool lazy_add) {
    utility::device_vector<Eigen::Vector2i> d_edges = edges;
    utility::device_vector<float> d_weights = weights;
    return AddEdges(d_edges, d_weights, lazy_add);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::AddEdges(
        const std::vector<Eigen::Vector2i> &edges,
        const std::vector<float> &weights,
        bool lazy_add) {
    utility::device_vector<Eigen::Vector2i> d_edges(edges.size());
    copy_host_to_device(edges, d_edges);
    utility::device_vector<float> d_weights(weights.size());
    copy_host_to_device(weights, d_weights);
    return AddEdges(d_edges, d_weights, lazy_add);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::RemoveEdge(const Eigen::Vector2i &edge) {
    bool has_colors = this->HasColors();
    bool has_weights = this->HasWeights();
    auto runs = [&edge, is_directed = is_directed_] (auto&... params) {
        remove_if_vectors(
                utility::exec_policy(0),
                check_edge_functor<typename std::remove_reference_t<decltype(params)>::value_type...>(
                        edge, is_directed),
                params...);
    };
    if (has_colors && has_weights) {
        runs(this->lines_, edge_weights_, this->colors_);
    } else if (has_colors && !has_weights) {
        runs(this->lines_, this->colors_);
    } else if (!has_colors && has_weights) {
        runs(this->lines_, edge_weights_);
    } else {
        runs(this->lines_);
    }
    return ConstructGraph(false);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::RemoveEdges(
        const utility::device_vector<Eigen::Vector2i> &edges) {
    bool has_colors = this->HasColors();
    bool has_weights = this->HasWeights();
    utility::device_vector<Eigen::Vector2i> new_lines(this->lines_.size());
    utility::device_vector<float> new_weights(edge_weights_.size());
    utility::device_vector<Eigen::Vector3f> new_colors(this->colors_.size());
    utility::device_vector<Eigen::Vector2i> sorted_edges = edges;
    utility::device_vector<Eigen::Vector2i> sorted_swap_edges = edges;
    thrust::sort(utility::exec_policy(0), sorted_edges.begin(),
                 sorted_edges.end());
    if (!is_directed_) {
        swap_index(sorted_swap_edges);
        thrust::sort(utility::exec_policy(0), sorted_swap_edges.begin(),
                     sorted_swap_edges.end());
    }
    auto cnst_w = thrust::make_constant_iterator<float>(1.0);
    auto cnst_c = thrust::make_constant_iterator<Eigen::Vector3f>(
            Eigen::Vector3f::Ones());
    auto runs = [is_directed = is_directed_] (auto& func,
                                              auto&& this_begins, auto&& this_ends,
                                              auto&& edge_begins, auto&& edge_ends,
                                              auto&& swap_edge_begins, auto&& swap_edge_ends,
                                              auto&... params) {
        auto begin = make_tuple_begin(params...);
        auto end1 = thrust::set_difference(
                this_begins, this_ends,
                edge_begins, edge_ends,
                begin, func);
        resize_all(thrust::distance(begin, end1), params...);
        if (!is_directed) {
            auto end2 = thrust::set_difference(
                    this_begins, this_ends,
                    swap_edge_begins, swap_edge_ends,
                    begin, func);
            resize_all(thrust::distance(begin, end2), params...);
        }
    };
    if (has_colors && has_weights) {
        auto func = tuple_element_compare_functor<
                EdgeWeightColor, 0, thrust::less<Eigen::Vector2i>>();
        runs(func,
             make_tuple_begin(this->lines_, edge_weights_, this->colors_),
             make_tuple_end(this->lines_, edge_weights_, this->colors_),
             make_tuple_iterator(sorted_edges.begin(), cnst_w, cnst_c),
             make_tuple_iterator(sorted_edges.end(), cnst_w, cnst_c),
             make_tuple_iterator(sorted_swap_edges.begin(), cnst_w, cnst_c),
             make_tuple_iterator(sorted_swap_edges.end(), cnst_w, cnst_c),
             new_lines, new_weights, new_colors);
    } else if (has_colors && !has_weights) {
        auto func = tuple_element_compare_functor<
                EdgeColor, 0, thrust::less<Eigen::Vector2i>>();
        runs(func,
             make_tuple_begin(this->lines_, this->colors_),
             make_tuple_end(this->lines_, this->colors_),
             make_tuple_iterator(sorted_edges.begin(), cnst_c),
             make_tuple_iterator(sorted_edges.end(), cnst_c),
             make_tuple_iterator(sorted_swap_edges.begin(), cnst_c),
             make_tuple_iterator(sorted_swap_edges.end(), cnst_c),
             new_lines, new_colors);
    } else if (!has_colors && has_weights) {
        auto func = tuple_element_compare_functor<
                EdgeWeight, 0, thrust::less<Eigen::Vector2i>>();
        runs(func,
             make_tuple_begin(this->lines_, edge_weights_),
             make_tuple_end(this->lines_, edge_weights_),
             make_tuple_iterator(sorted_edges.begin(), cnst_w),
             make_tuple_iterator(sorted_edges.end(), cnst_w),
             make_tuple_iterator(sorted_swap_edges.begin(), cnst_w),
             make_tuple_iterator(sorted_swap_edges.end(), cnst_w),
             new_lines, new_weights);
    } else {
        auto func = tuple_element_compare_functor<
                thrust::tuple<Eigen::Vector2i>, 0, thrust::less<Eigen::Vector2i>>();
        runs(func,
             make_tuple_begin(this->lines_),
             make_tuple_end(this->lines_),
             make_tuple_iterator(sorted_edges.begin()),
             make_tuple_iterator(sorted_edges.end()),
             make_tuple_iterator(sorted_swap_edges.begin()),
             make_tuple_iterator(sorted_swap_edges.end()),
             new_lines);
    }
    thrust::swap(this->lines_, new_lines);
    thrust::swap(edge_weights_, new_weights);
    thrust::swap(this->colors_, new_colors);
    return ConstructGraph(false);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::RemoveEdges(
        const thrust::host_vector<Eigen::Vector2i> &edges) {
    utility::device_vector<Eigen::Vector2i> d_edges = edges;
    return RemoveEdges(d_edges);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::RemoveEdges(const std::vector<Eigen::Vector2i> &edges) {
    utility::device_vector<Eigen::Vector2i> d_edges(edges.size());
    copy_host_to_device(edges, d_edges);
    return RemoveEdges(d_edges);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::PaintEdgeColor(const Eigen::Vector2i &edge,
                                       const Eigen::Vector3f &color) {
    if (!this->HasColors()) {
        this->colors_.resize(this->lines_.size(), Eigen::Vector3f::Ones());
    }
    thrust::transform_if(
            this->colors_.begin(), this->colors_.end(), this->lines_.begin(),
            this->colors_.begin(),
            [color] __device__(const Eigen::Vector3f &c) { return color; },
            [edge, is_directed = is_directed_] __device__(
                    const Eigen::Vector2i &line) {
                return line == edge ||
                       (!is_directed &&
                        line == Eigen::Vector2i(edge[1], edge[0]));
            });
    return *this;
}

template <int Dim>
Graph<Dim> &Graph<Dim>::PaintEdgesColor(
        const utility::device_vector<Eigen::Vector2i> &edges,
        const Eigen::Vector3f &color) {
    utility::device_vector<Eigen::Vector2i> sorted_edges = edges;
    utility::device_vector<size_t> indices(edges.size());
    thrust::sort(utility::exec_policy(0), sorted_edges.begin(),
                 sorted_edges.end());
    thrust::set_intersection_by_key(this->lines_.begin(), this->lines_.end(),
                                    sorted_edges.begin(), sorted_edges.end(),
                                    thrust::make_counting_iterator<size_t>(0),
                                    thrust::make_discard_iterator(),
                                    indices.begin());
    thrust::for_each(thrust::make_permutation_iterator(this->colors_.begin(),
                                                       indices.begin()),
                     thrust::make_permutation_iterator(this->colors_.begin(),
                                                       indices.end()),
                     [color] __device__(Eigen::Vector3f & c) { c = color; });
    if (!is_directed_) {
        swap_index(sorted_edges);
        thrust::sort(utility::exec_policy(0), sorted_edges.begin(),
                     sorted_edges.end());
        thrust::set_intersection_by_key(this->lines_.begin(), this->lines_.end(),
                                        sorted_edges.begin(), sorted_edges.end(),
                                        thrust::make_counting_iterator<size_t>(0),
                                        thrust::make_discard_iterator(),
                                        indices.begin());
        thrust::for_each(
                thrust::make_permutation_iterator(this->colors_.begin(),
                                                  indices.begin()),
                thrust::make_permutation_iterator(this->colors_.begin(),
                                                  indices.end()),
                [color] __device__(Eigen::Vector3f & c) { c = color; });
    }
    return *this;
}

template <int Dim>
Graph<Dim> &Graph<Dim>::PaintEdgesColor(
        const thrust::host_vector<Eigen::Vector2i> &edges,
        const Eigen::Vector3f &color) {
    utility::device_vector<Eigen::Vector2i> d_edges = edges;
    return PaintEdgesColor(d_edges, color);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::PaintEdgesColor(
        const std::vector<Eigen::Vector2i> &edges,
        const Eigen::Vector3f &color) {
    utility::device_vector<Eigen::Vector2i> d_edges(edges.size());
    copy_host_to_device(edges, d_edges);
    return PaintEdgesColor(d_edges, color);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::PaintNodeColor(int node, const Eigen::Vector3f &color) {
    if (!HasNodeColors()) {
        node_colors_.resize(this->points_.size(), Eigen::Vector3f::Ones());
    }
    node_colors_[node] = color;
    return *this;
}

template <int Dim>
Graph<Dim> &Graph<Dim>::PaintNodesColor(
        const utility::device_vector<int> &nodes,
        const Eigen::Vector3f &color) {
    if (!HasNodeColors()) {
        node_colors_.resize(this->points_.size(), Eigen::Vector3f::Ones());
    }
    thrust::for_each(node_colors_.begin(), node_colors_.end(),
                     [color] __device__(Eigen::Vector3f & c) { c = color; });
    return *this;
}

template <int Dim>
Graph<Dim> &Graph<Dim>::PaintNodesColor(const thrust::host_vector<int> &nodes,
                                        const Eigen::Vector3f &color) {
    utility::device_vector<int> d_nodes = nodes;
    return PaintNodesColor(d_nodes, color);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::PaintNodesColor(const std::vector<int> &nodes,
                                        const Eigen::Vector3f &color) {
    utility::device_vector<int> d_nodes(nodes.size());
    copy_host_to_device(nodes, d_nodes);
    return PaintNodesColor(d_nodes, color);
}

template <int Dim>
Graph<Dim> &Graph<Dim>::SetEdgeWeightsFromDistance() {
    edge_weights_.resize(this->lines_.size());
    thrust::transform(thrust::make_permutation_iterator(
                              this->points_.begin(),
                              thrust::make_transform_iterator(
                                      this->lines_.begin(),
                                      element_get_functor<Eigen::Vector2i, 0>())),
                      thrust::make_permutation_iterator(
                              this->points_.begin(),
                              thrust::make_transform_iterator(
                                      this->lines_.end(),
                                      element_get_functor<Eigen::Vector2i, 0>())),
                      thrust::make_permutation_iterator(
                              this->points_.begin(),
                              thrust::make_transform_iterator(
                                      this->lines_.begin(),
                                      element_get_functor<Eigen::Vector2i, 1>())),
                      edge_weights_.begin(),
                      [] __device__(const Eigen::Matrix<float, Dim, 1> &pt1,
                                    const Eigen::Matrix<float, Dim, 1> &pt2) {
                          return (pt1 - pt2).norm();
                      });
    return *this;
}

template <int Dim>
Graph<Dim> &Graph<Dim>::SetEdgeWeights(const utility::device_vector<Eigen::Vector2i> &edges, float weight) {
    utility::device_vector<Eigen::Vector2i> sorted_edges = edges;
    utility::device_vector<float> new_edge_weights = edge_weights_;
    new_edge_weights.resize(this->lines_.size(), 1.0);
    thrust::sort(utility::exec_policy(0), sorted_edges.begin(),
                 sorted_edges.end());
    utility::device_vector<size_t> result_indices(edges.size());
    auto end = thrust::set_intersection_by_key(this->lines_.begin(), this->lines_.end(),
                                               sorted_edges.begin(), sorted_edges.end(),
                                               thrust::make_counting_iterator<size_t>(0),
                                               thrust::make_discard_iterator(),
                                               result_indices.begin());
    result_indices.resize(thrust::distance(result_indices.begin(), end.second));
    auto begin = thrust::make_constant_iterator(weight);
    thrust::scatter(begin, begin + result_indices.size(),
                    result_indices.begin(), new_edge_weights.begin());
    if (!is_directed_) {
        swap_index(sorted_edges);
        thrust::sort(utility::exec_policy(0), sorted_edges.begin(),
                     sorted_edges.end());
        result_indices.resize(edges.size());
        auto end = thrust::set_intersection_by_key(this->lines_.begin(), this->lines_.end(),
                                                   sorted_edges.begin(), sorted_edges.end(),
                                                   thrust::make_counting_iterator<size_t>(0),
                                                   thrust::make_discard_iterator(),
                                                   result_indices.begin());
        result_indices.resize(thrust::distance(result_indices.begin(), end.second));
        auto begin = thrust::make_constant_iterator(weight);
        thrust::scatter(begin, begin + result_indices.size(),
                        result_indices.begin(), new_edge_weights.begin());
    }
    thrust::swap(edge_weights_, new_edge_weights);
    return *this;
}

template <int Dim>
std::shared_ptr<typename Graph<Dim>::SSSPResultArray> Graph<Dim>::DijkstraPaths(
        int start_node_index, int end_node_index) const {
    auto out = std::make_shared<typename Graph<Dim>::SSSPResultArray>();
    out->resize(this->points_.size());

    if (!IsConstructed()) {
        utility::LogError("[DijkstraPath] this graph is not constructed.");
        return out;
    }

    utility::device_vector<Eigen::Vector2i> sorted_lines = this->lines_;
    utility::device_vector<int> new_to_old_edge_table(this->lines_.size());
    utility::device_vector<int> old_to_new_edge_table(this->lines_.size());
    thrust::sequence(new_to_old_edge_table.begin(), new_to_old_edge_table.end(),
                     0);
    thrust::sort_by_key(utility::exec_policy(0), sorted_lines.begin(),
                        sorted_lines.end(), new_to_old_edge_table.begin(),
                        [] __device__(const Eigen::Vector2i &lhs,
                                      const Eigen::Vector2i &rhs) {
                            return lhs[1] < rhs[1];
                        });
    thrust::scatter(thrust::make_counting_iterator<size_t>(0),
                    thrust::make_counting_iterator(this->lines_.size()),
                    new_to_old_edge_table.begin(),
                    old_to_new_edge_table.begin());
    utility::device_vector<int> open_flags(this->points_.size(), 0);
    utility::device_vector<size_t> indices(this->points_.size());
    thrust::sequence(indices.begin(), indices.end(), 0);
    SSSPResultArray res_tmp(this->lines_.size());
    SSSPResultArray res_tmp_s(this->points_.size());
    open_flags[start_node_index] = 1;
    (*out)[start_node_index] = SSSPResult(0.0, start_node_index);
    relax_functor<Dim> func1(
            thrust::raw_pointer_cast(this->lines_.data()),
            thrust::raw_pointer_cast(edge_index_offsets_.data()),
            thrust::raw_pointer_cast(edge_weights_.data()),
            thrust::raw_pointer_cast(old_to_new_edge_table.data()),
            thrust::raw_pointer_cast(open_flags.data()),
            thrust::raw_pointer_cast(out->data()),
            thrust::raw_pointer_cast(res_tmp.data()));
    update_shortest_distances_functor<Dim> func2(
            thrust::raw_pointer_cast(open_flags.data()),
            thrust::raw_pointer_cast(out->data()),
            thrust::raw_pointer_cast(res_tmp_s.data()));
    compare_path_length_functor<Dim> func3(
            thrust::raw_pointer_cast(out->data()),
            thrust::raw_pointer_cast(open_flags.data()), end_node_index);
    size_t nt = this->points_.size();
    while (thrust::find(open_flags.begin(), open_flags.end(), 1) !=
           open_flags.end()) {
        if (end_node_index >= 0 &&
            thrust::count_if(utility::exec_policy(0), indices.begin(),
                             indices.begin() + nt, func3) == 0)
            break;
        thrust::for_each(indices.begin(), indices.begin() + nt, func1);
        const auto begin = thrust::make_transform_iterator(
                sorted_lines.begin(), element_get_functor<Eigen::Vector2i, 1>());
        auto end = thrust::reduce_by_key(
                utility::exec_policy(0), begin,
                begin + sorted_lines.size(), res_tmp.begin(), indices.begin(),
                res_tmp_s.begin(), thrust::equal_to<int>(),
                [] __device__(const SSSPResult &lhs, const SSSPResult &rhs) {
                    return (lhs.shortest_distance_ <= rhs.shortest_distance_)
                                   ? lhs
                                   : rhs;
                });
        nt = thrust::distance(indices.begin(), end.first);
        thrust::for_each(indices.begin(), indices.begin() + nt, func2);
    }
    return out;
}

template <int Dim>
std::shared_ptr<typename Graph<Dim>::SSSPResultHostArray>
Graph<Dim>::DijkstraPathsHost(int start_node_index, int end_node_index) const {
    auto out = DijkstraPaths(start_node_index, end_node_index);
    auto h_out = std::make_shared<typename Graph<Dim>::SSSPResultHostArray>();
    *h_out = *out;
    return h_out;
}

template <int Dim>
std::pair<std::shared_ptr<thrust::host_vector<int>>, float> Graph<Dim>::DijkstraPath(
        int start_node_index, int end_node_index) const {
    auto res = DijkstraPaths(start_node_index, end_node_index);
    SSSPResultHostArray h_res = *res;
    auto path_nodes = std::make_shared<thrust::host_vector<int>>();
    if (h_res[end_node_index].prev_index_ < 0) return std::make_pair(path_nodes, std::numeric_limits<float>::infinity());
    path_nodes->push_back(end_node_index);
    int prev_index = h_res[end_node_index].prev_index_;
    while (prev_index != start_node_index) {
        path_nodes->push_back(prev_index);
        prev_index = h_res[prev_index].prev_index_;
    }
    path_nodes->push_back(start_node_index);
    thrust::reverse(path_nodes->begin(), path_nodes->end());
    return std::make_pair(path_nodes, h_res[end_node_index].shortest_distance_);
}

template class Graph<2>;
template class Graph<3>;

}  // namespace geometry
}  // namespace cupoch