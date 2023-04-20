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
#include "cupoch/geometry/boundingvolume.h"
#include "cupoch/geometry/densegrid.inl"
#include "cupoch/geometry/distancetransform.h"
#include "cupoch/geometry/voxelgrid.h"
#include "cupoch/utility/platform.h"

#define BLOCKSIZE 8

namespace cupoch {
namespace geometry {

namespace {

__device__ const unsigned short INVALID_DENSE_GRID_INDEX =
        std::numeric_limits<unsigned short>::max();

struct flood_z_functor {
    flood_z_functor(const DistanceVoxel* input,
                    DistanceVoxel* output,
                    int resolution)
        : input_(input), output_(output), resolution_(resolution){};
    const DistanceVoxel* input_;
    DistanceVoxel* output_;
    const int resolution_;
    __device__ void operator()(size_t idx) {
        int x = idx / resolution_;
        int y = idx % resolution_;
        DistanceVoxel v1;
        for (int z = 0; z < resolution_; ++z) {
            int id = IndexOf(x, y, z, resolution_);
            DistanceVoxel v2 = input_[id];
            if (!v2.IsNotSite()) {
                v1 = v2;
            }
            output_[id] = v1;
        }
        for (int i = resolution_ - 2; i >= 0; --i) {
            int id = IndexOf(x, y, i, resolution_);
            unsigned short nz = v1.IsNotSite() ? INVALID_DENSE_GRID_INDEX
                                               : v1.nearest_index_[2];
            unsigned short dist1 = abs(nz - i);
            DistanceVoxel v2 = output_[id];
            nz = v2.IsNotSite() ? INVALID_DENSE_GRID_INDEX
                                : v2.nearest_index_[2];
            unsigned short dist2 = abs(nz - i);
            if (dist2 < dist1) {
                v1 = v2;
            }
            output_[id] = v1;
        }
    }
};

__device__ bool dominate(int x_1,
                         int y_1,
                         int z_1,
                         int x_2,
                         int y_2,
                         int z_2,
                         int x_3,
                         int y_3,
                         int z_3,
                         int x_0,
                         int z_0) {
    int k_1 = y_2 - y_1;
    int k_2 = y_3 - y_2;
    return (((y_1 + y_2) * k_1 + ((x_2 - x_1) * (x_1 + x_2 - (x_0 << 1)) +
                                  (z_2 - z_1) * (z_1 + z_2 - (z_0 << 1)))) *
                    k_2 >
            ((y_2 + y_3) * k_2 + ((x_3 - x_2) * (x_2 + x_3 - (x_0 << 1)) +
                                  (z_3 - z_2) * (z_2 + z_3 - (z_0 << 1)))) *
                    k_1);
}

struct maurer_axis_functor {
    maurer_axis_functor(const DistanceVoxel* input,
                        DistanceVoxel* output,
                        int resolution)
        : input_(input), output_(output), resolution_(resolution){};
    const DistanceVoxel* input_;
    DistanceVoxel* output_;
    const int resolution_;
    __device__ void operator()(size_t idx) {
        int x = idx / resolution_;
        int z = idx % resolution_;
        int lasty = 0;
        int flag = 0;
        DistanceVoxel s1;
        DistanceVoxel s2;
        DistanceVoxel p;
        int i = 0;
        for (; i < resolution_; ++i) {
            int id = IndexOf(x, i, z, resolution_);
            p = input_[id];
            if (!p.IsNotSite()) {
                while (s2.CheckHasNext()) {
                    if (!dominate(s1.nearest_index_[0], s2.nearest_index_[1],
                                  s1.nearest_index_[2], s2.nearest_index_[0],
                                  lasty, s2.nearest_index_[2],
                                  p.nearest_index_[0], i, p.nearest_index_[2],
                                  x, z)) {
                        break;
                    }
                    lasty = s2.nearest_index_[1];
                    s2 = s1;
                    s2.nearest_index_[1] = s1.nearest_index_[1];
                    if (s2.CheckHasNext()) {
                        s1 = output_[IndexOf(x, s2.nearest_index_[1], z,
                                             resolution_)];
                    }
                }
                s1 = s2;
                s2.nearest_index_ = Eigen::Vector3ui16(
                        p.nearest_index_[0], lasty, p.nearest_index_[2]);
                s2.state_ = flag;
                lasty = i;
                output_[id] = s2;
                flag = 1;
            }
        }

        if (p.IsNotSite()) {
            output_[IndexOf(x, i - 1, z, resolution_)] =
                    DistanceVoxel(Eigen::Vector3ui16(0, lasty, 0),
                                  DistanceVoxel::NotSite | flag);
        }
    }
};

__global__ void color_axis_kernel(const DistanceVoxel* input,
                                  DistanceVoxel* output,
                                  int resolution) {
    __shared__ DistanceVoxel block[BLOCKSIZE][BLOCKSIZE];
    int col = threadIdx.x;
    int tid = threadIdx.y;
    int tx = blockIdx.x * blockDim.x + col;
    int tz = blockIdx.y;
    int lasty = resolution - 1;
    DistanceVoxel last1;
    DistanceVoxel last2 = input[IndexOf(tx, lasty, tz, resolution)];
    if (last2.IsNotSite()) {
        lasty = last2.nearest_index_[1];
        if (last2.CheckHasNext()) {
            last2 = input[IndexOf(tx, lasty, tz, resolution)];
        }
    }
    if (last2.CheckHasNext()) {
        last1 = input[IndexOf(tx, last2.nearest_index_[1], tz, resolution)];
    }

    int n_step = resolution / blockDim.x;
    for (int step = 0; step < n_step; ++step) {
        int y_start = resolution - step * blockDim.x - 1;
        int y_end = resolution - (step + 1) * blockDim.x;
        for (int ty = y_start - tid; ty >= y_end; ty -= blockDim.y) {
            int dx = last2.nearest_index_[0] - tx;
            int dy = lasty - ty;
            int dz = last2.nearest_index_[2] - tz;
            int best = dx * dx + dy * dy + dz * dz;
            while (last2.CheckHasNext()) {
                dx = last1.nearest_index_[0] - tx;
                dy = last2.nearest_index_[1] - ty;
                dz = last1.nearest_index_[2] - tz;
                int dist = dx * dx + dy * dy + dz * dz;
                if (dist > best) break;
                best = dist;
                lasty = last2.nearest_index_[1];
                last2 = last1;
                if (last2.CheckHasNext()) {
                    last1 = input[IndexOf(tx, last2.nearest_index_[1], tz,
                                          resolution)];
                }
            }
            block[threadIdx.x][ty - y_end] = DistanceVoxel(
                    Eigen::Vector3ui16(lasty, last2.nearest_index_[0],
                                       last2.nearest_index_[2]),
                    last2.state_ & DistanceVoxel::State::NotSite);
        }
        __syncthreads();
        if (!threadIdx.y) {
            int id = IndexOf(y_end + threadIdx.x, blockIdx.x * blockDim.x, tz,
                             resolution);
            for (int i = 0; i < blockDim.x; ++i, id += resolution) {
                output[id] = block[i][threadIdx.x];
            }
        }
        __syncthreads();
    }
};

struct set_points_functor {
    set_points_functor(DistanceVoxel* voxels, int resolution)
        : voxels_(voxels), resolution_(resolution){};
    DistanceVoxel* voxels_;
    const int resolution_;
    __device__ void operator()(const Eigen::Vector3i& idxs) {
        int i = IndexOf(idxs, resolution_);
        voxels_[i] = DistanceVoxel(idxs.cast<unsigned short>(), 0);
    }
};

struct compute_distance_functor {
    compute_distance_functor(DistanceVoxel* voxels,
                             float voxel_size,
                             int resolution)
        : voxels_(voxels), voxel_size_(voxel_size), resolution_(resolution){};
    DistanceVoxel* voxels_;
    const float voxel_size_;
    const int resolution_;
    __device__ void operator()(size_t idx) {
        int x = idx / (resolution_ * resolution_);
        int yz = idx % (resolution_ * resolution_);
        int y = yz / resolution_;
        int z = yz % resolution_;
        auto diff = voxels_[idx].nearest_index_.cast<int>() -
                    Eigen::Vector3i(x, y, z);
        voxels_[idx].distance_ = diff.cast<float>().norm() * voxel_size_;
    }
};

struct compute_obstacle_cells_functor {
    compute_obstacle_cells_functor(float voxel_size,
                                   int resolution,
                                   const Eigen::Vector3f& origin1,
                                   const Eigen::Vector3f& origin2)
        : voxel_size_(voxel_size),
          resolution_(resolution),
          origin1_(origin1),
          origin2_(origin2){};
    const float voxel_size_;
    const int resolution_;
    const Eigen::Vector3f origin1_;
    const Eigen::Vector3f origin2_;
    __device__ Eigen::Vector3i operator()(const Eigen::Vector3i& key) const {
        Eigen::Vector3f abs_pos = key.cast<float>() * voxel_size_ + origin1_;
        return Eigen::device_vectorize<float, 3, ::floor>((abs_pos - origin2_) /
                                                          voxel_size_)
                       .cast<int>() +
               Eigen::Vector3i::Constant(resolution_ / 2);
    };
};

struct get_index_functor {
    get_index_functor(float voxel_size, int resolution, const Eigen::Vector3f& origin)
    : voxel_size_(voxel_size), resolution_(resolution), origin_(origin) {};
    const float voxel_size_;
    const int resolution_;
    const Eigen::Vector3f origin_;
    __device__ int operator()(const Eigen::Vector3f& query) const {
        Eigen::Vector3f qv =
                (query - origin_ +
                 0.5 * voxel_size_ * Eigen::Vector3f::Constant(resolution_)) /
                voxel_size_;
        Eigen::Vector3i idx =
                Eigen::device_vectorize<float, 3, ::floor>(qv.array())
                        .cast<int>();
        return IndexOf(idx, resolution_);
    };
};

}  // namespace

template class DenseGrid<DistanceVoxel>;

DistanceTransform::DistanceTransform()
    : DenseGrid<DistanceVoxel>(Geometry::GeometryType::DistanceTransform,
                               0.05,
                               512,
                               Eigen::Vector3f::Zero()) {
    buffer_.resize(voxels_.size());
}

DistanceTransform::DistanceTransform(float voxel_size,
                                     size_t resolution,
                                     const Eigen::Vector3f& origin)
    : DenseGrid<DistanceVoxel>(Geometry::GeometryType::DistanceTransform,
                               voxel_size,
                               resolution,
                               origin) {
    buffer_.resize(voxels_.size());
}

DistanceTransform::DistanceTransform(const DistanceTransform& other)
    : DenseGrid<DistanceVoxel>(Geometry::GeometryType::DistanceTransform,
                               other.voxel_size_,
                               other.resolution_,
                               other.origin_) {
    buffer_.resize(voxels_.size());
}

DistanceTransform::~DistanceTransform() {}

DistanceTransform& DistanceTransform::Reconstruct(float voxel_size,
                                                  size_t resolution) {
    DenseGrid::Reconstruct(voxel_size, resolution);
    buffer_.resize(voxels_.size());
    return *this;
}

DistanceTransform& DistanceTransform::ComputeEDT(
        const utility::device_vector<Eigen::Vector3i>& points) {
    ComputeVoronoiDiagram(points);
    compute_distance_functor func(thrust::raw_pointer_cast(voxels_.data()),
                                  voxel_size_, resolution_);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(voxels_.size()), func);
    return *this;
}

DistanceTransform& DistanceTransform::ComputeEDT(const VoxelGrid& voxelgrid) {
    if (std::abs(voxel_size_ - voxelgrid.voxel_size_) >
        std::numeric_limits<float>::epsilon()) {
        utility::LogError(
                "Unsupport computing Voronoi diagrams from different voxel "
                "size.");
        return *this;
    }
    utility::device_vector<Eigen::Vector3i> obs_cells(
            voxelgrid.voxels_keys_.size());
    compute_obstacle_cells_functor func(voxel_size_, resolution_,
                                        voxelgrid.origin_, origin_);
    thrust::transform(voxelgrid.voxels_keys_.begin(),
                      voxelgrid.voxels_keys_.end(), obs_cells.begin(), func);
    return ComputeEDT(obs_cells);
}

DistanceTransform& DistanceTransform::ComputeVoronoiDiagram(
        const utility::device_vector<Eigen::Vector3i>& points) {
    set_points_functor func0(thrust::raw_pointer_cast(buffer_.data()),
                             resolution_);
    thrust::for_each(points.begin(), points.end(), func0);

    flood_z_functor func1(thrust::raw_pointer_cast(buffer_.data()),
                          thrust::raw_pointer_cast(voxels_.data()),
                          resolution_);
    maurer_axis_functor func2(thrust::raw_pointer_cast(voxels_.data()),
                              thrust::raw_pointer_cast(buffer_.data()),
                              resolution_);

    thrust::for_each(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(resolution_ * resolution_),
            func1);
    thrust::for_each(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(resolution_ * resolution_),
            func2);

    dim3 block1 = dim3(BLOCKSIZE, 2);
    dim3 grid1 = dim3(resolution_ / block1.x, resolution_);
    color_axis_kernel<<<grid1, block1>>>(
            thrust::raw_pointer_cast(buffer_.data()),
            thrust::raw_pointer_cast(voxels_.data()), resolution_);
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());

    thrust::for_each(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(resolution_ * resolution_),
            func2);

    dim3 block2 = dim3(BLOCKSIZE, 2);
    dim3 grid2 = dim3(resolution_ / block2.x, resolution_);
    color_axis_kernel<<<grid2, block2>>>(
            thrust::raw_pointer_cast(buffer_.data()),
            thrust::raw_pointer_cast(voxels_.data()), resolution_);
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaGetLastError());

    return *this;
}

DistanceTransform& DistanceTransform::ComputeVoronoiDiagram(
        const VoxelGrid& voxelgrid) {
    if (std::abs(voxel_size_ - voxelgrid.voxel_size_) >
        std::numeric_limits<float>::epsilon()) {
        utility::LogError(
                "Unsupport computing Voronoi diagrams from different voxel "
                "size.");
        return *this;
    }
    utility::device_vector<Eigen::Vector3i> obs_cells(
            voxelgrid.voxels_keys_.size());
    compute_obstacle_cells_functor func(voxel_size_, resolution_,
                                        voxelgrid.origin_, origin_);
    thrust::transform(voxelgrid.voxels_keys_.begin(),
                      voxelgrid.voxels_keys_.end(), obs_cells.begin(), func);
    return ComputeVoronoiDiagram(obs_cells);
}

float DistanceTransform::GetDistance(const Eigen::Vector3f& query) const {
    Eigen::Vector3f qv =
            (query - origin_ +
             0.5 * voxel_size_ * Eigen::Vector3f::Constant(resolution_)) /
            voxel_size_;
    Eigen::Vector3i idx = (Eigen::floor(qv.array())).cast<int>();
    DistanceVoxel v = voxels_[IndexOf(idx, resolution_)];
    return v.distance_;
}

std::unique_ptr<utility::device_vector<float>> DistanceTransform::GetDistances(
        const utility::device_vector<Eigen::Vector3f>& queries) const {
    auto func = get_index_functor(voxel_size_, resolution_, origin_);
    auto dists = std::make_unique<utility::device_vector<float>>(queries.size());
    thrust::transform(
            thrust::make_permutation_iterator(
                    voxels_.begin(),
                    thrust::make_transform_iterator(queries.begin(), func)),
            thrust::make_permutation_iterator(
                    voxels_.begin(),
                    thrust::make_transform_iterator(queries.end(), func)),
            dists->begin(),
            [] __device__(const DistanceVoxel& v) { return v.distance_; });
    return dists;
}

}  // namespace geometry
}  // namespace cupoch