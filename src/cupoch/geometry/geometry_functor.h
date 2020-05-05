#pragma once

#include <Eigen/Core>

namespace cupoch {
namespace geometry {

struct compute_grid_center_functor {
    compute_grid_center_functor(float voxel_size,
                                const Eigen::Vector3f &origin)
        : voxel_size_(voxel_size),
          origin_(origin),
          half_voxel_size_(0.5 * voxel_size, 0.5 * voxel_size, 0.5 * voxel_size) {};
    const float voxel_size_;
    const Eigen::Vector3f origin_;
    const Eigen::Vector3f half_voxel_size_;
    __device__ Eigen::Vector3f operator()(const Eigen::Vector3i &x) const {
        return x.cast<float>() * voxel_size_ + origin_ + half_voxel_size_;
    }
};

template<typename T, int Size, int Index>
struct extract_element_functor {
    __device__ T operator() (const Eigen::Matrix<T, Size, 1>& x) { return x[Index]; };
};

template<typename T>
struct reverse_index_functor {
    __device__ Eigen::Matrix<T, 2, 1> operator() (const Eigen::Matrix<T, 2, 1>& x) {
        return Eigen::Matrix<T, 2, 1>(x[1], x[0]);
    };
};

template <typename TupleType, int Index, typename Func>
struct tuple_element_compare_functor {
    __device__ bool operator() (const TupleType& rhs, const TupleType& lhs) {
        return Func()(thrust::get<Index>(rhs), thrust::get<Index>(lhs));
    }
};

}
}