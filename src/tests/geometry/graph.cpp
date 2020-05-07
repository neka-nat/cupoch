#include "cupoch/geometry/graph.h"

#include "tests/test_utility/raw.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

TEST(Graph, Constructor) {
    geometry::Graph gp;

    // inherited from Geometry2D
    EXPECT_EQ(geometry::Geometry::GeometryType::Graph, gp.GetGeometryType());
    EXPECT_EQ(3, gp.Dimension());

    // public member variables
    EXPECT_EQ(0u, gp.points_.size());
    EXPECT_EQ(0u, gp.lines_.size());
    EXPECT_EQ(0u, gp.colors_.size());
    EXPECT_EQ(0u, gp.edge_index_offsets_.size());
    EXPECT_EQ(0u, gp.edge_weights_.size());

    // public members
    EXPECT_TRUE(gp.IsEmpty());

    ExpectEQ(Zero3f, gp.GetMinBound());
    ExpectEQ(Zero3f, gp.GetMaxBound());

    EXPECT_FALSE(gp.HasPoints());
    EXPECT_FALSE(gp.HasLines());
    EXPECT_FALSE(gp.HasColors());
    EXPECT_FALSE(gp.HasWeights());
    EXPECT_FALSE(gp.IsConstructed());
}

TEST(Graph, AddEdge) {
    geometry::Graph gp;
    thrust::host_vector<Eigen::Vector3f> points;
    points.push_back({0.0, 0.0, 0.0});
    points.push_back({1.0, 0.0, 0.0});
    points.push_back({0.0, 1.0, 0.0});
    points.push_back({1.0, 1.0, 0.0});
    points.push_back({2.0, 2.0, 0.0});
    gp.SetPoints(points);
    gp.AddEdge({0, 1});
    gp.AddEdge({0, 2});
    gp.AddEdge({1, 3});
    gp.AddEdge({2, 3});
    gp.AddEdge({3, 4});

    EXPECT_EQ(gp.lines_.size(), 10);
    thrust::host_vector<int> edge_index_offsets = gp.GetEdgeIndexOffsets();
    EXPECT_EQ(edge_index_offsets.size(), 6);
    EXPECT_EQ(edge_index_offsets[0], 0);
    EXPECT_EQ(edge_index_offsets[1], 2);
    EXPECT_EQ(edge_index_offsets[2], 4);
    EXPECT_EQ(edge_index_offsets[3], 6);
    EXPECT_EQ(edge_index_offsets[4], 9);
    EXPECT_EQ(edge_index_offsets[5], 10);
}

TEST(Graph, DijkstraPath) {
    geometry::Graph gp;
    thrust::host_vector<Eigen::Vector3f> points;
    points.push_back({0.0, 0.0, 0.0});
    points.push_back({1.0, 0.0, 0.0});
    points.push_back({0.0, 1.0, 0.0});
    points.push_back({1.0, 1.0, 0.0});
    points.push_back({2.0, 2.0, 0.0});
    gp.SetPoints(points);
    gp.AddEdge({0, 1});
    gp.AddEdge({0, 2}, 2.0);
    gp.AddEdge({1, 3});
    gp.AddEdge({2, 3});
    gp.AddEdge({3, 4});

    auto res = gp.DijkstraPathsHost(0);
    EXPECT_EQ(res.size(), 5);
    EXPECT_EQ(res[1].shortest_distance_, 1.0);
    EXPECT_EQ(res[2].shortest_distance_, 2.0);
    EXPECT_EQ(res[3].shortest_distance_, 2.0);
    EXPECT_EQ(res[4].shortest_distance_, 3.0);
}