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
#include "cupoch/geometry/graph.h"

#include "tests/test_utility/raw.h"
#include "tests/test_utility/unit_test.h"

using namespace Eigen;
using namespace cupoch;
using namespace std;
using namespace unit_test;

TEST(Graph, Constructor) {
    geometry::Graph<3> gp;

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
    geometry::Graph<3> gp;
    std::vector<Eigen::Vector3f> points;
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
    std::vector<int> edge_index_offsets = gp.GetEdgeIndexOffsets();
    EXPECT_EQ(edge_index_offsets.size(), 6);
    EXPECT_EQ(edge_index_offsets[0], 0);
    EXPECT_EQ(edge_index_offsets[1], 2);
    EXPECT_EQ(edge_index_offsets[2], 4);
    EXPECT_EQ(edge_index_offsets[3], 6);
    EXPECT_EQ(edge_index_offsets[4], 9);
    EXPECT_EQ(edge_index_offsets[5], 10);
}

TEST(Graph, DijkstraPath) {
    geometry::Graph<3> gp;
    std::vector<Eigen::Vector3f> points;
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
    EXPECT_EQ(res->size(), 5);
    EXPECT_EQ((*res)[1].shortest_distance_, 1.0);
    EXPECT_EQ((*res)[2].shortest_distance_, 2.0);
    EXPECT_EQ((*res)[3].shortest_distance_, 2.0);
    EXPECT_EQ((*res)[4].shortest_distance_, 3.0);
}