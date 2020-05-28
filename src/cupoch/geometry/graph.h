#pragma once

#include "cupoch/geometry/lineset.h"

namespace cupoch {
namespace geometry {

class Graph : public LineSet {
public:
    struct SSSPResult {
        __host__ __device__ SSSPResult(float shortest_distance = std::numeric_limits<float>::infinity(),
                                       int prev_index = -1)
                                       : shortest_distance_(shortest_distance), prev_index_(prev_index) {};
        __host__ __device__ ~SSSPResult() {};
        __host__ __device__ SSSPResult(const SSSPResult& other)
        : shortest_distance_(other.shortest_distance_), prev_index_(other.prev_index_) {};
        float shortest_distance_;
        int prev_index_;
    };
    typedef utility::device_vector<SSSPResult> SSSPResultArray;
    typedef thrust::host_vector<SSSPResult> SSSPResultHostArray;

    typedef thrust::tuple<Eigen::Vector2i, float, Eigen::Vector3f> EdgeWeightColor;
    typedef thrust::tuple<Eigen::Vector2i, Eigen::Vector3f> EdgeColor;
    typedef thrust::tuple<Eigen::Vector2i, float> EdgeWeight;

    Graph();
    Graph(const utility::device_vector<Eigen::Vector3f> &points);
    Graph(const thrust::host_vector<Eigen::Vector3f> &points);
    Graph(const Graph& other);
    ~Graph();

    thrust::host_vector<int> GetEdgeIndexOffsets() const;
    void SetEdgeIndexOffsets(const thrust::host_vector<int>& edge_index_offsets);
    thrust::host_vector<float> GetEdgeWeights() const;
    void SetEdgeWeights(const thrust::host_vector<float>& edge_weights);

    __host__ __device__
    bool HasWeights() const {
        return edge_weights_.size() > 0 && lines_.size() == edge_weights_.size();
    };

    __host__ __device__
    bool HasNodeColors() const {
        return node_colors_.size() > 0 && points_.size() == node_colors_.size();
    };

    __host__ __device__
    bool IsConstructed() const {
        return edge_index_offsets_.size() == points_.size() + 1 && lines_.size() == edge_weights_.size();
    }

    Graph &Clear() override;

    Graph &ConstructGraph(bool set_edge_weights_from_distance = true);
    Graph &ConnectToNearestNeighbors(float max_edge_distance, int max_num_edges = 30);
    Graph &AddNodeAndConnect(const Eigen::Vector3f& point, float max_edge_distance = 0.0f, bool lazy_add = false);

    Graph &AddEdge(const Eigen::Vector2i &edge, float weight = 1.0, bool lazy_add = false);
    Graph &AddEdges(const utility::device_vector<Eigen::Vector2i> &edges,
                    const utility::device_vector<float> &weights = utility::device_vector<float>(),
                    bool lazy_add = false);
    Graph &AddEdges(const thrust::host_vector<Eigen::Vector2i> &edges,
                    const thrust::host_vector<float> &weights = thrust::host_vector<float>(),
                    bool lazy_add = false);

    Graph &RemoveEdge(const Eigen::Vector2i &edge);
    Graph &RemoveEdges(const utility::device_vector<Eigen::Vector2i> &edges);
    Graph &RemoveEdges(const thrust::host_vector<Eigen::Vector2i> &edges);

    Graph &PaintEdgeColor(const Eigen::Vector2i &edge, const Eigen::Vector3f &color);
    Graph &PaintEdgesColor(const utility::device_vector<Eigen::Vector2i> &edges, const Eigen::Vector3f &color);
    Graph &PaintEdgesColor(const thrust::host_vector<Eigen::Vector2i> &edges, const Eigen::Vector3f &color);

    Graph &PaintNodeColor(int node, const Eigen::Vector3f &color);
    Graph &PaintNodesColor(const utility::device_vector<int> &nodes, const Eigen::Vector3f &color);
    Graph &PaintNodesColor(const thrust::host_vector<int> &nodes, const Eigen::Vector3f &color);

    Graph &SetEdgeWeightsFromDistance();

    std::shared_ptr<SSSPResultArray> DijkstraPaths(int start_node_index, int end_node_index = -1) const;
    std::shared_ptr<SSSPResultHostArray> DijkstraPathsHost(int start_node_index, int end_node_index = -1) const;
    std::shared_ptr<thrust::host_vector<int>> DijkstraPath(int start_node_index, int end_node_index) const;

    static std::shared_ptr<Graph> CreateFromTriangleMesh(const TriangleMesh &input);
    static std::shared_ptr<Graph> CreateFromAxisAlignedBoundingBox(const geometry::AxisAlignedBoundingBox& bbox,
                                                                   const Eigen::Vector3i& resolutions);
    static std::shared_ptr<Graph> CreateFromAxisAlignedBoundingBox(const Eigen::Vector3f& min_bound,
                                                                   const Eigen::Vector3f& max_bound,
                                                                   const Eigen::Vector3i& resolutions);

public:
    utility::device_vector<int> edge_index_offsets_;
    utility::device_vector<float> edge_weights_;
    utility::device_vector<Eigen::Vector3f> node_colors_;
    bool is_directed_ = false;
};

}
}