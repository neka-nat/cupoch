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
#pragma once
#include "cupoch/geometry/densegrid.h"
#include "cupoch/utility/eigen.h"

namespace cupoch {

namespace geometry {
class PointCloud;
class VoxelGrid;

class OccupancyVoxel {
public:
    __host__ __device__ OccupancyVoxel() {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i& grid_index)
        : grid_index_(grid_index.cast<unsigned short>()) {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i& grid_index,
                                       float prob_log)
        : grid_index_(grid_index.cast<unsigned short>()), prob_log_(prob_log) {}
    __host__ __device__ OccupancyVoxel(const Eigen::Vector3i& grid_index,
                                       float prob_log,
                                       const Eigen::Vector3f& color)
        : grid_index_(grid_index.cast<unsigned short>()),
          color_(color),
          prob_log_(prob_log) {}
    __host__ __device__ ~OccupancyVoxel() {}

public:
    Eigen::Vector3ui16 grid_index_ = Eigen::Vector3ui16::Zero();
    Eigen::Vector3f color_ = Eigen::Vector3f(0.0, 0.0, 1.0);
    float prob_log_ = std::numeric_limits<float>::quiet_NaN();
};

inline OccupancyVoxel operator+(const OccupancyVoxel& lhs,
                                const OccupancyVoxel& rhs) {
    OccupancyVoxel out = lhs;
    out.prob_log_ += rhs.prob_log_;
    out.color_ += rhs.color_;
    out.color_ *= 0.5;
    return out;
}

inline OccupancyVoxel operator-(const OccupancyVoxel& lhs,
                                const OccupancyVoxel& rhs) {
    OccupancyVoxel out = lhs;
    out.prob_log_ -= rhs.prob_log_;
    out.color_ += Eigen::Vector3f::Ones() - rhs.color_;
    out.color_ *= 0.5;
    return out;
}

class OccupancyGrid : public DenseGrid<OccupancyVoxel> {
public:
    OccupancyGrid();
    OccupancyGrid(float voxel_size,
                  size_t resolution = 512,
                  const Eigen::Vector3f& origin = Eigen::Vector3f::Zero());
    ~OccupancyGrid();
    OccupancyGrid(const OccupancyGrid& other);

    OccupancyGrid& Clear() override;

    Eigen::Vector3f GetMinBound() const override;
    Eigen::Vector3f GetMaxBound() const override;

    bool HasVoxels() const { return voxels_.size() > 0; }
    bool HasColors() const {
        return true;  // By default, the colors are (1.0, 1.0, 1.0)
    }
    bool IsOccupied(const Eigen::Vector3f& point) const;
    bool IsUnknown(const Eigen::Vector3f& point) const;
    thrust::tuple<bool, OccupancyVoxel> GetVoxel(
            const Eigen::Vector3f& point) const;
    std::shared_ptr<utility::device_vector<OccupancyVoxel>> ExtractKnownVoxels()
            const;
    std::shared_ptr<utility::device_vector<OccupancyVoxel>> ExtractFreeVoxels()
            const;
    std::shared_ptr<utility::device_vector<OccupancyVoxel>>
    ExtractOccupiedVoxels() const;

    OccupancyGrid& Reconstruct(float voxel_size, int resolution);
    OccupancyGrid& SetFreeArea(const Eigen::Vector3f& min_bound,
                               const Eigen::Vector3f& max_bound);

    OccupancyGrid& Insert(const utility::device_vector<Eigen::Vector3f>& points,
                          const Eigen::Vector3f& viewpoint,
                          float max_range = -1.0);
    OccupancyGrid& Insert(
            const utility::pinned_host_vector<Eigen::Vector3f>& points,
            const Eigen::Vector3f& viewpoint,
            float max_range = -1.0);
    OccupancyGrid& Insert(
            const thrust::host_vector<Eigen::Vector3f>& points,
            const Eigen::Vector3f& viewpoint,
            float max_range = -1.0);
    OccupancyGrid& Insert(const std::vector<Eigen::Vector3f>& points,
                          const Eigen::Vector3f& viewpoint,
                          float max_range = -1.0);
    OccupancyGrid& Insert(const PointCloud& pointcloud,
                          const Eigen::Vector3f& viewpoint,
                          float max_range = -1.0);

    OccupancyGrid& AddVoxel(const Eigen::Vector3i& voxels,
                            bool occupied = false);
    OccupancyGrid& AddVoxels(
            const utility::device_vector<Eigen::Vector3i>& voxels,
            bool occupied = false);

    static std::shared_ptr<OccupancyGrid> CreateFromVoxelGrid(
            const VoxelGrid& input);

private:
    template <typename Func>
    std::shared_ptr<utility::device_vector<OccupancyVoxel>> ExtractBoundVoxels(
            Func func) const;

public:
    Eigen::Vector3ui16 min_bound_ = Eigen::Vector3ui16::Zero();
    Eigen::Vector3ui16 max_bound_ = Eigen::Vector3ui16::Zero();
    float clamping_thres_min_ = -2.0f;
    float clamping_thres_max_ = 3.5f;
    float prob_hit_log_ = 0.85f;
    float prob_miss_log_ = -0.4f;
    float occ_prob_thres_log_ = 0.0f;
    bool visualize_free_area_ = true;
};

}  // namespace geometry

}  // namespace cupoch