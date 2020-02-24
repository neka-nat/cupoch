#pragma once
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include <Eigen/Core>
#include <string>
#include <vector>

#include "cupoch/utility/device_vector.h"
#include "cupoch/utility/platform.h"

namespace thrust {

template <int Dim>
struct equal_to<Eigen::Matrix<int, Dim, 1>> {
    typedef Eigen::Matrix<int, Dim, 1> first_argument_type;
    typedef Eigen::Matrix<int, Dim, 1> second_argument_type;
    typedef bool result_type;
    // clang-format off
    __thrust_exec_check_disable__
    __host__ __device__ bool operator()(
            const Eigen::Matrix<int, Dim, 1> &lhs,
            const Eigen::Matrix<int, Dim, 1> &rhs) const {
        for (int i = 0; i < Dim; ++i) {
            if (lhs[i] != rhs[i]) return false;
        }
        return true;
    }
    // clang-format on
};

template <typename MatType, typename VecType>
struct plus<thrust::tuple<MatType, VecType, float>> {
    __host__ __device__ thrust::tuple<MatType, VecType, float> operator()(
            const thrust::tuple<MatType, VecType, float> &x,
            const thrust::tuple<MatType, VecType, float> &y) const {
        MatType mat = thrust::get<0>(x) + thrust::get<0>(y);
        VecType vec = thrust::get<1>(x) + thrust::get<1>(y);
        float r = thrust::get<2>(x) + thrust::get<2>(y);
        return thrust::make_tuple(mat, vec, r);
    }
};

}  // namespace thrust

namespace Eigen {

template <int Dim>
__host__ __device__ bool operator<(const Eigen::Matrix<int, Dim, 1> &lhs,
                                   const Eigen::Matrix<int, Dim, 1> &rhs) {
    for (int i = 0; i < Dim; ++i) {
        if (lhs[i] != rhs[i]) return lhs[i] < rhs[i];
    }
    return false;
}

__host__ __device__ inline bool operator==(const Eigen::Vector3i &lhs,
                                           const Eigen::Vector3i &rhs) {
    return (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2]);
}

__host__ __device__ inline bool operator!=(const Eigen::Vector3i &lhs,
                                           const Eigen::Vector3i &rhs) {
    return (lhs[0] != rhs[0] || lhs[1] != rhs[1] || lhs[2] != rhs[2]);
}

__host__ __device__ inline bool operator!=(const Eigen::Vector3f &lhs,
                                           const Eigen::Vector3f &rhs) {
    return (lhs[0] != rhs[0] || lhs[1] != rhs[1] || lhs[2] != rhs[2]);
}

template <typename ArrayType>
__host__ __device__ bool device_any(const ArrayType &array) {
    for (int i = 0; i < array.size(); ++i) {
        if (array[i]) return true;
    }
    return false;
}

}  // namespace Eigen

namespace cupoch {

template <class... Args>
struct add_tuple_functor
    : public thrust::binary_function<const thrust::tuple<Args...>,
                                     const thrust::tuple<Args...>,
                                     thrust::tuple<Args...>> {
    __host__ __device__ thrust::tuple<Args...> operator()(
            const thrust::tuple<Args...> &x,
            const thrust::tuple<Args...> &y) const;
};

template <class... Args>
struct devided_tuple_functor
    : public thrust::binary_function<const thrust::tuple<Args...>,
                                     const int,
                                     thrust::tuple<Args...>> {
    __host__ __device__ thrust::tuple<Args...> operator()(
            const thrust::tuple<Args...> &x, const int &y) const;
};

template <class T1>
struct add_tuple_functor<T1>
    : public thrust::binary_function<const thrust::tuple<T1>,
                                     const thrust::tuple<T1>,
                                     thrust::tuple<T1>> {
    __host__ __device__ thrust::tuple<T1> operator()(
            const thrust::tuple<T1> &x, const thrust::tuple<T1> &y) const {
        thrust::tuple<T1> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) + thrust::get<0>(y);
        return ans;
    }
};

template <class T1, class T2>
struct add_tuple_functor<T1, T2>
    : public thrust::binary_function<const thrust::tuple<T1, T2>,
                                     const thrust::tuple<T1, T2>,
                                     thrust::tuple<T1, T2>> {
    __host__ __device__ thrust::tuple<T1, T2> operator()(
            const thrust::tuple<T1, T2> &x,
            const thrust::tuple<T1, T2> &y) const {
        thrust::tuple<T1, T2> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) + thrust::get<0>(y);
        thrust::get<1>(ans) = thrust::get<1>(x) + thrust::get<1>(y);
        return ans;
    }
};

template <class T1, class T2, class T3>
struct add_tuple_functor<T1, T2, T3>
    : public thrust::binary_function<const thrust::tuple<T1, T2, T3>,
                                     const thrust::tuple<T1, T2, T3>,
                                     thrust::tuple<T1, T2, T3>> {
    __host__ __device__ thrust::tuple<T1, T2, T3> operator()(
            const thrust::tuple<T1, T2, T3> &x,
            const thrust::tuple<T1, T2, T3> &y) const {
        thrust::tuple<T1, T2, T3> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) + thrust::get<0>(y);
        thrust::get<1>(ans) = thrust::get<1>(x) + thrust::get<1>(y);
        thrust::get<2>(ans) = thrust::get<2>(x) + thrust::get<2>(y);
        return ans;
    }
};

template <class T1>
struct devided_tuple_functor<T1>
    : public thrust::binary_function<const thrust::tuple<T1>,
                                     const int,
                                     thrust::tuple<T1>> {
    __host__ __device__ thrust::tuple<T1> operator()(const thrust::tuple<T1> &x,
                                                     const int &y) const {
        thrust::tuple<T1> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) / static_cast<float>(y);
        return ans;
    }
};

template <class T1, class T2>
struct devided_tuple_functor<T1, T2>
    : public thrust::binary_function<const thrust::tuple<T1, T2>,
                                     const int,
                                     thrust::tuple<T1, T2>> {
    __host__ __device__ thrust::tuple<T1, T2> operator()(
            const thrust::tuple<T1, T2> &x, const int &y) const {
        thrust::tuple<T1, T2> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) / static_cast<float>(y);
        thrust::get<1>(ans) = thrust::get<1>(x) / static_cast<float>(y);
        return ans;
    }
};

template <class T1, class T2, class T3>
struct devided_tuple_functor<T1, T2, T3>
    : public thrust::binary_function<const thrust::tuple<T1, T2, T3>,
                                     const int,
                                     thrust::tuple<T1, T2, T3>> {
    __host__ __device__ thrust::tuple<T1, T2, T3> operator()(
            const thrust::tuple<T1, T2, T3> &x, const int &y) const {
        thrust::tuple<T1, T2, T3> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) / static_cast<float>(y);
        thrust::get<1>(ans) = thrust::get<1>(x) / static_cast<float>(y);
        thrust::get<2>(ans) = thrust::get<2>(x) / static_cast<float>(y);
        return ans;
    }
};

template <class... Args>
thrust::zip_iterator<thrust::tuple<Args...>> make_tuple_iterator(
        const Args &... args) {
    return thrust::make_zip_iterator(thrust::make_tuple(args...));
}

template <class... Args>
thrust::zip_iterator<thrust::tuple<Args...>> make_tuple_iterator(
        Args &... args) {
    return thrust::make_zip_iterator(thrust::make_tuple(args...));
}

namespace utility {

/// Function to split a string, mimics boost::split
/// http://stackoverflow.com/questions/236129/split-a-string-in-c
void SplitString(std::vector<std::string> &tokens,
                 const std::string &str,
                 const std::string &delimiters = " ",
                 bool trim_empty_str = true);

/// String util: find length of current word staring from a position
/// By default, alpha numeric chars and chars in valid_chars are considered
/// as valid charactors in a word
size_t WordLength(const std::string &doc,
                  size_t start_pos,
                  const std::string &valid_chars = "_");

std::string &LeftStripString(std::string &str,
                             const std::string &chars = "\t\n\v\f\r ");

std::string &RightStripString(std::string &str,
                              const std::string &chars = "\t\n\v\f\r ");

/// Strip empty charactors in front and after string. Similar to Python's
/// str.strip()
std::string &StripString(std::string &str,
                         const std::string &chars = "\t\n\v\f\r ");

template <typename T>
void CopyToDeviceMultiStream(const thrust::host_vector<T> &src,
                             utility::device_vector<T> &dst,
                             int n_stream = MAX_NUM_STREAMS) {
    const int step = src.size() / n_stream;
    int step_size = step * sizeof(T);
    for (int i = 0; i < n_stream; ++i) {
        const int offset = i * step;
        if (i == n_stream - 1)
            step_size = (src.size() - step * (n_stream - 1)) * sizeof(T);
        cudaMemcpyAsync(thrust::raw_pointer_cast(&dst[offset]),
                        thrust::raw_pointer_cast(&src[offset]), step_size,
                        cudaMemcpyHostToDevice, GetStream(i));
    }
}

template <typename T>
void CopyFromDeviceMultiStream(const utility::device_vector<T> &src,
                               thrust::host_vector<T> &dst,
                               int n_stream = MAX_NUM_STREAMS) {
    const int step = src.size() / n_stream;
    int step_size = step * sizeof(T);
    for (int i = 0; i < n_stream; ++i) {
        const int offset = i * step;
        if (i == n_stream - 1)
            step_size = (src.size() - step * (n_stream - 1)) * sizeof(T);
        cudaMemcpyAsync(thrust::raw_pointer_cast(&dst[offset]),
                        thrust::raw_pointer_cast(&src[offset]), step_size,
                        cudaMemcpyDeviceToHost, GetStream(i));
    }
}

}  // namespace utility
}  // namespace cupoch