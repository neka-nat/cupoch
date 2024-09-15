/**
 * Copyright (c) 2022 Neka-Nat
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
#pragma once
#include <Eigen/Core>

#include "cupoch/utility/device_vector.h"

namespace lbvh {
struct BVHNode;
}

namespace cupoch {

namespace geometry {
class Geometry;
}

namespace knn {

using AABB = std::pair<Eigen::Vector3f, Eigen::Vector3f>;


class LinearBoundingVolumeHierarchyKNN {
public:
    LinearBoundingVolumeHierarchyKNN(size_t leaf_size = 32, bool compact = false, bool sort_queries = false, bool shrink_to_fit = false);
    LinearBoundingVolumeHierarchyKNN(const utility::device_vector<Eigen::Vector3f> &data, size_t leaf_size = 32, bool compact = false, bool sort_queries = false, bool shrink_to_fit = false);
    LinearBoundingVolumeHierarchyKNN(const std::vector<Eigen::Vector3f> &data, size_t leaf_size = 32, bool compact = false, bool sort_queries = false, bool shrink_to_fit = false);
    ~LinearBoundingVolumeHierarchyKNN();
    LinearBoundingVolumeHierarchyKNN(const LinearBoundingVolumeHierarchyKNN &) = delete;
    LinearBoundingVolumeHierarchyKNN &operator=(const LinearBoundingVolumeHierarchyKNN &) = delete;

public:
    template <typename InputIterator, int Dim>
    int SearchNN(InputIterator first,
                 InputIterator last,
                 float radius,
                 utility::device_vector<unsigned int> &indices,
                 utility::device_vector<float> &distance2) const;

    template <typename T>
    int SearchNN(const utility::device_vector<T> &query,
                 float radius,
                 utility::device_vector<unsigned int> &indices,
                 utility::device_vector<float> &distance2) const;

    template <typename T>
    int SearchNN(const T &query,
                 float radius,
                 thrust::host_vector<unsigned int> &indices,
                 thrust::host_vector<float> &distance2) const;

    template <typename T>
    int SearchNN(const T &query,
                 float radius,
                 std::vector<unsigned int> &indices,
                 std::vector<float> &distance2) const;

    template <typename T>
    bool SetRawData(const utility::device_vector<T> &data);

private:
    size_t n_points_;
    size_t n_nodes_;
    size_t leaf_size_;
    size_t dimension_ = 0;
    bool compact_;
    bool sort_queries_;
    bool shrink_to_fit_;

    AABB extent_;
    std::unique_ptr<utility::device_vector<lbvh::BVHNode>> nodes_;
    utility::device_vector<float3> data_float3_;
    utility::device_vector<unsigned int> sorted_indices_;
    unsigned int root_node_index_;
};

}
}

