#pragma once
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/type_traits/integer_sequence.h>

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

template<typename VectorType>
struct elementwise_minimum {
    __device__ VectorType operator()(const VectorType &a,
                                     const VectorType &b) {
        return a.array().min(b.array()).matrix();
    }
};

template<typename VectorType>
struct elementwise_maximum {
    __device__ VectorType operator()(const VectorType &a,
                                     const VectorType &b) {
        return a.array().max(b.array()).matrix();
    }
};

}  // namespace thrust

namespace Eigen {

template <typename T, int Dim>
__host__ __device__ bool operator<(const Eigen::Matrix<T, Dim, 1> &lhs,
                                   const Eigen::Matrix<T, Dim, 1> &rhs) {
    for (int i = 0; i < Dim; ++i) {
        if (lhs[i] != rhs[i]) return lhs[i] < rhs[i];
    }
    return false;
}

template <typename T, int Dim>
__host__ __device__ bool operator>(const Eigen::Matrix<T, Dim, 1> &lhs,
                                   const Eigen::Matrix<T, Dim, 1> &rhs) {
    for (int i = 0; i < Dim; ++i) {
        if (lhs[i] != rhs[i]) return lhs[i] > rhs[i];
    }
    return false;
}

template <typename T>
__host__ __device__ inline bool operator==(const Eigen::Matrix<T, 2, 1> &lhs,
                                           const Eigen::Matrix<T, 2, 1> &rhs) {
    return (lhs[0] == rhs[0] && lhs[1] == rhs[1]);
}

template <typename T>
__host__ __device__ inline bool operator!=(const Eigen::Matrix<T, 2, 1> &lhs,
                                           const Eigen::Matrix<T, 2, 1> &rhs) {
    return (lhs[0] != rhs[0] || lhs[1] != rhs[1]);
}

template <typename T>
__host__ __device__ inline bool operator==(const Eigen::Matrix<T, 3, 1> &lhs,
                                           const Eigen::Matrix<T, 3, 1> &rhs) {
    return (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2]);
}

template <typename T>
__host__ __device__ inline bool operator!=(const Eigen::Matrix<T, 3, 1> &lhs,
                                           const Eigen::Matrix<T, 3, 1> &rhs) {
    return (lhs[0] != rhs[0] || lhs[1] != rhs[1] || lhs[2] != rhs[2]);
}

template <typename ArrayType>
__host__ __device__ bool device_any(const ArrayType &array) {
    for (int i = 0; i < array.size(); ++i) {
        if (array[i]) return true;
    }
    return false;
}

template <typename T, int Dim, float (*Func)(float)>
__host__ __device__ Eigen::Matrix<T, Dim, 1> device_vectorize(const Eigen::Matrix<T, Dim, 1>& x) {
    Eigen::Matrix<T, Dim, 1> ans;
    for (int i = 0; i < Dim; ++i) {
        ans[i] = Func(x[i]);
    }
    return ans;
}

}  // namespace Eigen

namespace cupoch {

template<typename T>
__host__ __device__ void add_fn(T& x, const T& y) {
    x += y;
}

template<typename T, std::size_t... Is>
__host__ __device__ void add_tuple_impl(T& t, const T& y, std::index_sequence<Is...>) {
    std::initializer_list<int>{((void)add_fn(thrust::get<Is>(t), thrust::get<Is>(y)), 0)...};
}

template <class... Args>
struct add_tuple_functor
    : public thrust::binary_function<const thrust::tuple<Args...>,
                                     const thrust::tuple<Args...>,
                                     thrust::tuple<Args...>> {
    __host__ __device__ thrust::tuple<Args...> operator()(
            const thrust::tuple<Args...> &x, const thrust::tuple<Args...> &y) const {
        thrust::tuple<Args...> ans = x;
        add_tuple_impl(ans, y, thrust::make_index_sequence<sizeof...(Args)>{});
        return ans;
    }
};

template<typename T>
__host__ __device__ void devide_fn(T& x, const int& y) {
    x /= static_cast<float>(y);
}

template<typename T, std::size_t... Is>
__host__ __device__ void devide_tuple_impl(T& t, const int& y, std::index_sequence<Is...>) {
    std::initializer_list<int>{((void)devide_fn(thrust::get<Is>(t), y), 0)...};
}

template <class... Args>
struct devide_tuple_functor
    : public thrust::binary_function<const thrust::tuple<Args...>,
                                     const int,
                                     thrust::tuple<Args...>> {
    __host__ __device__ thrust::tuple<Args...> operator()(
            const thrust::tuple<Args...> &x, const int &y) const {
        thrust::tuple<Args...> ans = x;
        devide_tuple_impl(ans, y, thrust::make_index_sequence<sizeof...(Args)>{});
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

template <class... Args>
auto make_tuple_begin(const Args &... args) {
    return make_tuple_iterator(std::begin(args)...);
}

template <class... Args>
auto make_tuple_begin(Args &... args) {
    return make_tuple_iterator(std::begin(args)...);
}

template <class... Args>
auto make_tuple_end(const Args &... args) {
    return make_tuple_iterator(std::end(args)...);
}

template <class... Args>
auto make_tuple_end(Args &... args) {
    return make_tuple_iterator(std::end(args)...);
}

template <class T>
void resize_fn(size_t new_size, T& a){ a.resize(new_size); }

template <class... Args>
void resize_all(size_t new_size, Args &... args) {
    std::initializer_list<int>{((void)resize_fn(new_size, args), 0)...};
}

__host__ __device__ inline int IndexOf(int x, int y, int z, int resolution) {
    return x * resolution * resolution + y * resolution + z;
}

__host__ __device__ inline int IndexOf(const Eigen::Vector3i &xyz, int resolution) {
    return IndexOf(xyz(0), xyz(1), xyz(2), resolution);
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