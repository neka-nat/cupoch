#include "cupoch/geometry/graph.h"
#include "cupoch/geometry/geometry_functor.h"

#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>

namespace cupoch {
namespace geometry {

namespace {

struct replace_color_functor {
    replace_color_functor(const Eigen::Vector2i* lines,
                          Eigen::Vector3f* colors,
                          const Eigen::Vector2i& edge,
                          const Eigen::Vector3f& color,
                          bool is_directed)
                          : lines_(lines), colors_(colors),
                          edge_(edge), color_(color),
                          is_directed_(is_directed) {};
    const Eigen::Vector2i* lines_;
    Eigen::Vector3f* colors_;
    const Eigen::Vector2i edge_;
    const Eigen::Vector3f color_;
    const bool is_directed_;
    __device__ void operator() (size_t idx) const {
        if (lines_[idx] == edge_ || (!is_directed_ && lines_[idx] == Eigen::Vector2i(edge_[1], edge_[0]))) {
            colors_[idx] = color_;
        }
    }
};

struct replace_colors_functor {
    replace_colors_functor(Eigen::Vector3f* colors,
                           const Eigen::Vector3f& color)
                           : colors_(colors),
                           color_(color) {};
    Eigen::Vector3f* colors_;
    const Eigen::Vector3f color_;
    __device__ void operator() (size_t idx) const {
        colors_[idx] = color_;
    }
};

struct relax_functor {
    relax_functor(const Eigen::Vector2i* lines,
                  const int* edge_index_offsets,
                  const float* edge_weights,
                  int* open_flags,
                  const Graph::SSSPResult* res,
                  Graph::SSSPResult* res_tmp)
                  : lines_(lines), edge_index_offsets_(edge_index_offsets),
                  edge_weights_(edge_weights), open_flags_(open_flags),
                  res_(res), res_tmp_(res_tmp) {};
    const Eigen::Vector2i* lines_;
    const int* edge_index_offsets_;
    const float* edge_weights_;
    int* open_flags_;
    const Graph::SSSPResult* res_;
    Graph::SSSPResult* res_tmp_;
    __device__ void operator() (size_t idx) {
        if (open_flags_[idx] == 0) return;
        open_flags_[idx] = 0;
        int s_edge = edge_index_offsets_[idx];
        int e_edge = edge_index_offsets_[idx + 1];
        for (int j = s_edge; j < e_edge; ++j) {
            int k = lines_[j][0];
            res_tmp_[j].shortest_distance_ = res_[k].shortest_distance_ + edge_weights_[j];
            res_tmp_[j].prev_index_ = k;
        }
    }
};

struct update_shortest_distances_functor {
    update_shortest_distances_functor(int* open_flags,
                                      Graph::SSSPResult* res,
                                      const Graph::SSSPResult* res_tmp)
                                      : open_flags_(open_flags),
                                      res_(res), res_tmp_(res_tmp) {};
    int* open_flags_;
    Graph::SSSPResult* res_;
    const Graph::SSSPResult* res_tmp_;
    __device__ void operator() (size_t idx) {
        if (res_[idx].shortest_distance_ > res_tmp_[idx].shortest_distance_) {
            res_[idx] = res_tmp_[idx];
            open_flags_[idx] = 1;
        }
    }
};

}

Graph::Graph() : LineSet(Geometry::GeometryType::Graph) {}
Graph::Graph(const utility::device_vector<Eigen::Vector3f> &points)
 : LineSet(Geometry::GeometryType::Graph, points, utility::device_vector<Eigen::Vector2i>()) {}
Graph::Graph(const thrust::host_vector<Eigen::Vector3f> &points)
 : LineSet(Geometry::GeometryType::Graph, points, utility::device_vector<Eigen::Vector2i>()) { ConstructGraph(); }
Graph::~Graph() {}
 Graph::Graph(const Graph &other)
 : LineSet(Geometry::GeometryType::Graph, other.points_, other.lines_),
 edge_index_offsets_(other.edge_index_offsets_), edge_weights_(other.edge_weights_),
 is_directed_(other.is_directed_) {}

thrust::host_vector<int> Graph::GetEdgeIndexOffsets() const {
    thrust::host_vector<int> edge_index_offsets = edge_index_offsets_;
    return edge_index_offsets;
}

void Graph::SetEdgeIndexOffsets(const thrust::host_vector<int>& edge_index_offsets) {
    edge_index_offsets_ = edge_index_offsets;
}

thrust::host_vector<float> Graph::GetEdgeWeights() const {
    thrust::host_vector<float> edge_weights = edge_weights_;
    return edge_weights;
}

void Graph::SetEdgeWeights(const thrust::host_vector<float>& edge_weights) {
    edge_weights_ = edge_weights;
}

Graph &Graph::Clear() {
    LineSet::Clear();
    edge_index_offsets_.clear();
    edge_weights_.clear();
    return *this;
}

Graph &Graph::ConstructGraph() {
    if (lines_.empty()) {
        utility::LogError("[ConstructGraph] Graph has no edges.");
        return *this;
    }

    bool has_colors = HasColors();
    bool has_weights = HasWeights();
    if (has_colors && has_weights) {
        thrust::sort_by_key(lines_.begin(), lines_.end(),
                            make_tuple_iterator(edge_weights_.begin(), colors_.begin()));
    } else if (!has_colors && has_weights) {
        thrust::sort_by_key(lines_.begin(), lines_.end(), edge_weights_.begin());
    } else if (has_colors && !has_weights) {
        thrust::sort_by_key(lines_.begin(), lines_.end(), colors_.begin());
    } else {
        thrust::sort(lines_.begin(), lines_.end());
        edge_weights_.resize(lines_.size(), 1.0);
    }
    edge_index_offsets_.resize(points_.size() + 1, 0);
    utility::device_vector<int> indices(lines_.size());
    utility::device_vector<int> counts(lines_.size());
    const auto begin = thrust::make_transform_iterator(lines_.begin(), extract_element_functor<int, 2, 0>());
    auto end = thrust::reduce_by_key(begin, begin + lines_.size(), thrust::make_constant_iterator<int>(1),
                                     indices.begin(), counts.begin());
    indices.resize(thrust::distance(indices.begin(), end.first));
    counts.resize(thrust::distance(counts.begin(), end.second));
    thrust::gather(indices.begin(), indices.end(), counts.begin(), edge_index_offsets_.begin());
    thrust::exclusive_scan(edge_index_offsets_.begin(), edge_index_offsets_.end(), edge_index_offsets_.begin());
    return *this;
}

Graph &Graph::AddEdge(const Eigen::Vector2i &edge, float weight) {
    lines_.push_back(edge);
    edge_weights_.push_back(weight);
    if (!is_directed_) {
        lines_.push_back(Eigen::Vector2i(edge[1], edge[0]));
        edge_weights_.push_back(weight);
    }
    if (HasColors()) {
        colors_.push_back(Eigen::Vector3f::Ones());
        if (!is_directed_) colors_.push_back(Eigen::Vector3f::Ones());
    }
    return ConstructGraph();
}

Graph &Graph::AddEdges(const utility::device_vector<Eigen::Vector2i> &edges,
                       const utility::device_vector<float> &weights) {
    if (!weights.empty() && edges.size() != weights.size()) {
        utility::LogError("[AddEdges] edges size is not equal to weights size.");
        return *this;
    }
    size_t n_old_lines = lines_.size();
    lines_.insert(lines_.end(), edges.begin(), edges.end());
    if (!is_directed_) {
        lines_.insert(lines_.end(), thrust::make_transform_iterator(edges.begin(), reverse_index_functor<int>()),
                      thrust::make_transform_iterator(edges.end(), reverse_index_functor<int>()));
    }
    if (weights.empty()) {
        if (!is_directed_) {
            edge_weights_.resize(2 * lines_.size());
        } else {
            edge_weights_.resize(lines_.size());
        }
        thrust::fill(edge_weights_.begin() + n_old_lines, edge_weights_.end(), 1.0);
    } else {
        edge_weights_.insert(edge_weights_.end(), weights.begin(), weights.end());
        if (!is_directed_) edge_weights_.insert(edge_weights_.end(), weights.begin(), weights.end());
    }
    if (HasColors()) {
        colors_.resize(lines_.size());
        thrust::fill(colors_.begin() + n_old_lines, colors_.end(), Eigen::Vector3f::Ones());
    }
    return ConstructGraph();
}

Graph &Graph::AddEdges(const thrust::host_vector<Eigen::Vector2i> &edges,
                       const thrust::host_vector<float> &weights) {
    utility::device_vector<Eigen::Vector2i> d_edges = edges;
    utility::device_vector<float> d_weights = weights;
    return AddEdges(d_edges, d_weights);
}

Graph &Graph::RemoveEdge(const Eigen::Vector2i &edge) {
    bool has_colors = HasColors();
    bool has_weights = HasWeights();
    if (has_colors && has_weights) {
        auto begin = make_tuple_iterator(lines_.begin(), edge_weights_.begin(), colors_.begin());
        auto end = thrust::remove_if(begin,
                make_tuple_iterator(lines_.end(), edge_weights_.end(), colors_.end()),
                [edge, is_directed = is_directed_] __device__ (const thrust::tuple<Eigen::Vector2i, float, Eigen::Vector3f> &x) {
                    const Eigen::Vector2i& l = thrust::get<0>(x);
                    return l == edge || (!is_directed && l == Eigen::Vector2i(edge[1], edge[0]));
                });
        size_t n_out = thrust::distance(begin, end);
        lines_.resize(n_out);
        edge_weights_.resize(n_out);
        colors_.resize(n_out);
    } else if (has_colors && !has_weights) {
        auto begin = make_tuple_iterator(lines_.begin(), colors_.begin());
        auto end = thrust::remove_if(begin,
                make_tuple_iterator(lines_.end(), colors_.end()),
                [edge, is_directed = is_directed_] __device__ (const thrust::tuple<Eigen::Vector2i, Eigen::Vector3f> &x) {
                    const Eigen::Vector2i& l = thrust::get<0>(x);
                    return l == edge || (!is_directed && l == Eigen::Vector2i(edge[1], edge[0]));
                });
        size_t n_out = thrust::distance(begin, end);
        lines_.resize(n_out);
        colors_.resize(n_out);
    } else if (!has_colors && has_weights) {
        auto begin = make_tuple_iterator(lines_.begin(), edge_weights_.begin());
        auto end = thrust::remove_if(begin,
                make_tuple_iterator(lines_.end(), edge_weights_.end()),
                [edge, is_directed = is_directed_] __device__ (const thrust::tuple<Eigen::Vector2i, float> &x) {
                    const Eigen::Vector2i& l = thrust::get<0>(x);
                    return l == edge || (!is_directed && l == Eigen::Vector2i(edge[1], edge[0]));
                });
        size_t n_out = thrust::distance(begin, end);
        lines_.resize(n_out);
        edge_weights_.resize(n_out);
    } else {
        auto end = thrust::remove_if(lines_.begin(), lines_.end(),
            [edge, is_directed = is_directed_] __device__ (const Eigen::Vector2i &l) {
                return l == edge || (!is_directed && l == Eigen::Vector2i(edge[1], edge[0]));
            });
        lines_.resize(thrust::distance(lines_.begin(), end));
    }
    return ConstructGraph();
}

Graph &Graph::RemoveEdges(const utility::device_vector<Eigen::Vector2i> &edges) {
    bool has_colors = HasColors();
    bool has_weights = HasWeights();
    utility::device_vector<Eigen::Vector2i> new_lines;
    utility::device_vector<float> new_weights;
    utility::device_vector<Eigen::Vector3f> new_colors;
    utility::device_vector<Eigen::Vector2i> sorted_edges = edges;
    thrust::sort(sorted_edges.begin(), sorted_edges.end());
    auto cnst_w = thrust::make_constant_iterator<float>(1.0);
    auto cnst_c = thrust::make_constant_iterator<Eigen::Vector3f>(Eigen::Vector3f::Ones());
    if (has_colors && has_weights) {
        auto func = tuple_element_compare_functor<thrust::tuple<Eigen::Vector2i, float, Eigen::Vector3f>, 0, thrust::greater<Eigen::Vector2i>>();
        auto begin = make_tuple_iterator(new_lines.begin(), new_weights.end(), new_colors.begin());
        auto end1 = thrust::set_difference(make_tuple_iterator(lines_.begin(), edge_weights_.begin(), colors_.begin()),
                make_tuple_iterator(lines_.end(), edge_weights_.end(), colors_.end()),
                make_tuple_iterator(sorted_edges.begin(), cnst_w, cnst_c),
                make_tuple_iterator(sorted_edges.end(), cnst_w, cnst_c),
                begin, func);
        size_t n_out1 = thrust::distance(begin, end1);
        new_lines.resize(n_out1);
        new_weights.resize(n_out1);
        new_colors.resize(n_out1);
        if (!is_directed_) {
            auto end2 = thrust::set_difference(make_tuple_iterator(lines_.begin(), edge_weights_.begin(), colors_.begin()),
                    make_tuple_iterator(lines_.end(), edge_weights_.end(), colors_.end()),
                    make_tuple_iterator(thrust::make_transform_iterator(sorted_edges.begin(), reverse_index_functor<int>()), cnst_w, cnst_c),
                    make_tuple_iterator(thrust::make_transform_iterator(sorted_edges.end(), reverse_index_functor<int>()), cnst_w, cnst_c),
                    begin, func);
            size_t n_out2 = thrust::distance(begin, end1);
            new_lines.resize(n_out2);
            new_weights.resize(n_out2);
            new_colors.resize(n_out2);
        }
    } else if (has_colors && !has_weights) {
        auto func = tuple_element_compare_functor<thrust::tuple<Eigen::Vector2i, Eigen::Vector3f>, 0, thrust::greater<Eigen::Vector2i>>();
        auto begin = make_tuple_iterator(new_lines.begin(), new_colors.end());
        auto end1 = thrust::set_difference(make_tuple_iterator(lines_.begin(), colors_.begin()),
                make_tuple_iterator(lines_.end(), colors_.end()),
                make_tuple_iterator(sorted_edges.begin(), cnst_c),
                make_tuple_iterator(sorted_edges.end(), cnst_c),
                begin, func);
        size_t n_out1 = thrust::distance(begin, end1);
        new_lines.resize(n_out1);
        new_colors.resize(n_out1);
        if (!is_directed_) {
            auto end2 = thrust::set_difference(make_tuple_iterator(lines_.begin(), colors_.begin()),
                    make_tuple_iterator(lines_.end(), colors_.end()),
                    make_tuple_iterator(thrust::make_transform_iterator(sorted_edges.begin(), reverse_index_functor<int>()), cnst_c),
                    make_tuple_iterator(thrust::make_transform_iterator(sorted_edges.end(), reverse_index_functor<int>()), cnst_c),
                    begin, func);
            size_t n_out2 = thrust::distance(begin, end1);
            new_lines.resize(n_out2);
            new_colors.resize(n_out2);
        }
    } else if (!has_colors && has_weights) {
        auto func = tuple_element_compare_functor<thrust::tuple<Eigen::Vector2i, float>, 0, thrust::greater<Eigen::Vector2i>>();
        auto begin = make_tuple_iterator(new_lines.begin(), new_weights.end());
        auto end1 = thrust::set_difference(make_tuple_iterator(lines_.begin(), edge_weights_.begin()),
                make_tuple_iterator(lines_.end(), edge_weights_.end()),
                make_tuple_iterator(sorted_edges.begin(), cnst_w),
                make_tuple_iterator(sorted_edges.end(), cnst_w),
                begin, func);
        size_t n_out1 = thrust::distance(begin, end1);
        new_lines.resize(n_out1);
        new_weights.resize(n_out1);
        if (!is_directed_) {
            auto end2 = thrust::set_difference(make_tuple_iterator(lines_.begin(), edge_weights_.begin()),
                    make_tuple_iterator(lines_.end(), edge_weights_.end()),
                    make_tuple_iterator(thrust::make_transform_iterator(sorted_edges.begin(), reverse_index_functor<int>()), cnst_w),
                    make_tuple_iterator(thrust::make_transform_iterator(sorted_edges.end(), reverse_index_functor<int>()), cnst_w),
                    begin, func);
            size_t n_out2 = thrust::distance(begin, end1);
            new_lines.resize(n_out2);
            new_weights.resize(n_out2);
        }
    } else {
        auto end1 = thrust::set_difference(lines_.begin(), lines_.end(),
                sorted_edges.begin(), sorted_edges.end(), new_lines.begin());
        new_lines.resize(thrust::distance(new_lines.begin(), end1));
        if (!is_directed_) {
            auto end2 = thrust::set_difference(lines_.begin(), lines_.end(),
                    thrust::make_transform_iterator(sorted_edges.begin(), reverse_index_functor<int>()),
                    thrust::make_transform_iterator(sorted_edges.end(), reverse_index_functor<int>()),
                    new_lines.begin());
            new_lines.resize(thrust::distance(new_lines.begin(), end2));
        }
    }
    lines_ = new_lines;
    edge_weights_ = new_weights;
    colors_ = new_colors;
    return ConstructGraph();
}

Graph &Graph::RemoveEdges(const thrust::host_vector<Eigen::Vector2i> &edges) {
    utility::device_vector<Eigen::Vector2i> d_edges = edges;
    return RemoveEdges(d_edges);
}

Graph &Graph::PaintEdgeColor(const Eigen::Vector2i &edge, const Eigen::Vector3f &color) {
    if (!HasColors()) {
        colors_.resize(lines_.size(), Eigen::Vector3f::Ones());
    }
    replace_color_functor func(thrust::raw_pointer_cast(lines_.data()),
                               thrust::raw_pointer_cast(colors_.data()),
                               edge, color, is_directed_);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(lines_.size()), func);
    return *this;
}

Graph &Graph::PaintEdgesColor(const utility::device_vector<Eigen::Vector2i> &edges, const Eigen::Vector3f &color) {
    utility::device_vector<Eigen::Vector2i> sorted_edges = edges;
    utility::device_vector<size_t> indices(edges.size());
    thrust::sort(sorted_edges.begin(), sorted_edges.end());
    thrust::set_intersection(make_tuple_iterator(lines_.begin(), thrust::make_counting_iterator<size_t>(0)),
            make_tuple_iterator(lines_.end(), thrust::make_counting_iterator(lines_.size())),
            make_tuple_iterator(sorted_edges.begin(), thrust::make_constant_iterator<size_t>(0)),
            make_tuple_iterator(sorted_edges.end(), thrust::make_constant_iterator<size_t>(0)),
            make_tuple_iterator(thrust::make_discard_iterator(), indices.begin()),
            tuple_element_compare_functor<thrust::tuple<Eigen::Vector2i, size_t>, 0, thrust::greater<Eigen::Vector2i>>());
    replace_colors_functor func(thrust::raw_pointer_cast(colors_.data()), color);
    thrust::for_each(indices.begin(), indices.end(), func);
    if (!is_directed_) {
        thrust::transform(sorted_edges.begin(), sorted_edges.end(), sorted_edges.begin(), reverse_index_functor<int>());
        thrust::sort(sorted_edges.begin(), sorted_edges.end());
        thrust::set_intersection(make_tuple_iterator(lines_.begin(), thrust::make_counting_iterator<size_t>(0)),
                make_tuple_iterator(lines_.end(), thrust::make_counting_iterator(lines_.size())),
                make_tuple_iterator(sorted_edges.begin(), thrust::make_constant_iterator<size_t>(0)),
                make_tuple_iterator(sorted_edges.end(), thrust::make_constant_iterator<size_t>(0)),
                make_tuple_iterator(thrust::make_discard_iterator(), indices.begin()),
                tuple_element_compare_functor<thrust::tuple<Eigen::Vector2i, size_t>, 0, thrust::greater<Eigen::Vector2i>>());
        thrust::for_each(indices.begin(), indices.end(), func);
    }
    return *this;
}

Graph &Graph::PaintEdgesColor(const thrust::host_vector<Eigen::Vector2i> &edges, const Eigen::Vector3f &color) {
    utility::device_vector<Eigen::Vector2i> d_edges = edges;
    return PaintEdgesColor(d_edges, color);
}

Graph &Graph::SetEdgeWeightsFromDistance() {
    edge_weights_.resize(lines_.size());
    Eigen::Vector3f *pt_ptr = thrust::raw_pointer_cast(points_.data());
    thrust::transform(lines_.begin(), lines_.end(), edge_weights_.begin(),
                      [pt_ptr] __device__ (const Eigen::Vector2i& edge) {
                          return (pt_ptr[edge[0]] - pt_ptr[edge[1]]).norm();
                      });
    return *this;
}

Graph::SSSPResultArray Graph::DijkstraPath(int start_node_index) const {
    SSSPResultArray out(points_.size());

    if (!IsConstructed()) {
        utility::LogError("[DijkstraPath] this graph is not constructed.");
        return out;
    }

    utility::device_vector<int> open_flags(points_.size(), 0);
    utility::device_vector<int> indices(lines_.size());
    utility::device_vector<Eigen::Vector2i> sorted_lines(lines_.size());
    SSSPResultArray res_tmp_line(lines_.size());
    SSSPResultArray sorted_res_tmp_line(lines_.size());
    SSSPResultArray sorted_res_tmp_line_s(lines_.size());
    SSSPResultArray res_tmp(points_.size());
    open_flags[start_node_index] = 1;
    out[start_node_index] = SSSPResult(0.0);
    relax_functor func1(thrust::raw_pointer_cast(lines_.data()),
                        thrust::raw_pointer_cast(edge_index_offsets_.data()),
                        thrust::raw_pointer_cast(edge_weights_.data()),
                        thrust::raw_pointer_cast(open_flags.data()),
                        thrust::raw_pointer_cast(out.data()),
                        thrust::raw_pointer_cast(res_tmp_line.data()));
    update_shortest_distances_functor func2(thrust::raw_pointer_cast(open_flags.data()),
                                            thrust::raw_pointer_cast(out.data()),
                                            thrust::raw_pointer_cast(res_tmp.data()));
    while (thrust::reduce(open_flags.begin(), open_flags.end()) > 0) {
        thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator(points_.size()), func1);
        sorted_lines = lines_;
        sorted_res_tmp_line = res_tmp_line;
        thrust::sort_by_key(sorted_lines.begin(), sorted_lines.end(), sorted_res_tmp_line.begin(),
                            [] __device__ (const Eigen::Vector2i &lhs, const Eigen::Vector2i &rhs) {
                                return lhs[1] < rhs[1];
                            });
        const auto begin = thrust::make_transform_iterator(sorted_lines.begin(), extract_element_functor<int, 2, 1>());
        auto end = thrust::reduce_by_key(begin, begin + sorted_lines.size(), sorted_res_tmp_line.begin(),
                                         indices.begin(), sorted_res_tmp_line_s.begin(),
                                         thrust::equal_to<int>(),
                                         [] __device__ (const SSSPResult& lhs, const SSSPResult& rhs) {
                                             return (lhs.shortest_distance_<= rhs.shortest_distance_) ? lhs : rhs;
                                         });
        size_t ns = thrust::distance(indices.begin(), end.first);
        thrust::gather(indices.begin(), indices.begin() + ns, sorted_res_tmp_line_s.begin(), res_tmp.begin());
        thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator(points_.size()), func2);
    }
    return out;
}

Graph::SSSPResultHostArray Graph::DijkstraPathHost(int start_node_index) const {
    auto out = DijkstraPath(start_node_index);
    SSSPResultHostArray h_out = out;
    return h_out;
}

thrust::host_vector<int> Graph::DijkstraPath(int start_node_index, int end_node_index) const {
    auto res = DijkstraPath(start_node_index);
    SSSPResultHostArray h_res = res;
    if (h_res[end_node_index].prev_index_ < 0) return thrust::host_vector<int>();
    thrust::host_vector<int> path_nodes;
    path_nodes.push_back(end_node_index);
    int prev_index = h_res[end_node_index].prev_index_;
    while (prev_index != start_node_index) {
        path_nodes.push_back(prev_index);
        prev_index = h_res[prev_index].prev_index_;
    }
    path_nodes.push_back(start_node_index);
    thrust::reverse(path_nodes.begin(), path_nodes.end());
    return path_nodes;
}

}
}