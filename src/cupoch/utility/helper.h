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
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#if THRUST_VERSION >= 100905 && !defined(_WIN32)
#include <thrust/type_traits/integer_sequence.h>
#else
namespace thrust {
template <std::size_t N>
using make_index_sequence = std::make_index_sequence<N>;
}
#endif

#include <stdgpu/functional.h>

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
    __host__ __device__ bool operator()(
            const Eigen::Matrix<int, Dim, 1> &lhs,
            const Eigen::Matrix<int, Dim, 1> &rhs) const {
        #pragma unroll
        for (int i = 0; i < Dim; ++i) {
            if (lhs[i] != rhs[i]) return false;
        }
        return true;
    }
    // clang-format on
};

namespace is_eigen_matrix_detail {
template <typename T>
std::true_type test(const Eigen::MatrixBase<T> *);
std::false_type test(...);
}  // namespace is_eigen_matrix_detail

template <typename T>
struct is_eigen_matrix
    : public decltype(is_eigen_matrix_detail::test(std::declval<T *>())) {};

template <typename VectorType, typename Enable = void>
struct elementwise_minimum;

template <typename VectorType, typename Enable = void>
struct elementwise_maximum;

template <typename VectorType>
struct elementwise_minimum<
        VectorType,
        typename std::enable_if<is_eigen_matrix<VectorType>::value>::type> {
    __device__ VectorType operator()(const VectorType &a, const VectorType &b) {
        return a.array().min(b.array()).matrix();
    }
};

template <typename VectorType>
struct elementwise_maximum<
        VectorType,
        typename std::enable_if<is_eigen_matrix<VectorType>::value>::type> {
    __device__ VectorType operator()(const VectorType &a, const VectorType &b) {
        return a.array().max(b.array()).matrix();
    }
};

struct discard_iterable {
    thrust::discard_iterator<> begin() { return thrust::make_discard_iterator(); };
    thrust::discard_iterator<> end() { return thrust::make_discard_iterator(); };
    thrust::discard_iterator<> begin() const { return thrust::make_discard_iterator(); };
    thrust::discard_iterator<> end() const { return thrust::make_discard_iterator(); };
    discard_iterable& operator++() { return *this; };
    discard_iterable operator++(int) { return *this; };
};

}  // namespace thrust

namespace Eigen {

template <typename T, int Dim>
__host__ __device__ bool operator<(const Eigen::Matrix<T, Dim, 1> &lhs,
                                   const Eigen::Matrix<T, Dim, 1> &rhs) {
#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        if (lhs[i] != rhs[i]) return lhs[i] < rhs[i];
    }
    return false;
}

template <typename T, int Dim>
__host__ __device__ bool operator>(const Eigen::Matrix<T, Dim, 1> &lhs,
                                   const Eigen::Matrix<T, Dim, 1> &rhs) {
#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        if (lhs[i] != rhs[i]) return lhs[i] > rhs[i];
    }
    return false;
}

template <typename T, int Dim>
__host__ __device__ inline bool operator==(
        const Eigen::Matrix<T, Dim, 1> &lhs,
        const Eigen::Matrix<T, Dim, 1> &rhs) {
#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

template <typename T, int Dim>
__host__ __device__ inline bool operator!=(
        const Eigen::Matrix<T, Dim, 1> &lhs,
        const Eigen::Matrix<T, Dim, 1> &rhs) {
#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        if (lhs[i] != rhs[i]) return true;
    }
    return false;
}

template <typename ArrayType>
__host__ __device__ bool device_all(const ArrayType &array) {
#pragma unroll
    for (int i = 0; i < ArrayType::SizeAtCompileTime; ++i) {
        if (!array[i]) return false;
    }
    return true;
}

template <typename ArrayType>
__host__ __device__ bool device_any(const ArrayType &array) {
#pragma unroll
    for (int i = 0; i < ArrayType::SizeAtCompileTime; ++i) {
        if (array[i]) return true;
    }
    return false;
}

template <typename T, int Dim, float (*Func)(float)>
__host__ __device__ Eigen::Matrix<T, Dim, 1> device_vectorize(
        const Eigen::Matrix<T, Dim, 1> &x) {
    Eigen::Matrix<T, Dim, 1> ans;
#pragma unroll
    for (int i = 0; i < Dim; ++i) {
        ans[i] = Func(x[i]);
    }
    return ans;
}

template <typename T, int M, int N, float (*Func)(float)>
__host__ __device__ Eigen::Matrix<T, M, N> device_vectorize(
        const Eigen::Matrix<T, M, N> &x) {
    Eigen::Matrix<T, M, N> ans;
#pragma unroll
    for (int k = 0; k < M * N; ++k) {
        int i = k / N;
        int j = i % N;
        ans(i, j) = Func(x(i, j));
    }
    return ans;
}

}  // namespace Eigen

namespace cupoch {

template <typename T>
__host__ __device__ void add_fn(T &x, const T &y) {
    x += y;
}

template <typename T, std::size_t... Is>
__host__ __device__ void add_tuple_impl(T &t,
                                        const T &y,
                                        thrust::integer_sequence<std::size_t, Is...>) {
    std::initializer_list<int>{
            ((void)add_fn(thrust::get<Is>(t), thrust::get<Is>(y)), 0)...};
}

template <class... Args>
struct add_tuple_functor
    : public thrust::binary_function<const thrust::tuple<Args...>,
                                     const thrust::tuple<Args...>,
                                     thrust::tuple<Args...>> {
    __host__ __device__ thrust::tuple<Args...> operator()(
            const thrust::tuple<Args...> &x,
            const thrust::tuple<Args...> &y) const {
        thrust::tuple<Args...> ans = x;
        add_tuple_impl(ans, y, thrust::make_index_sequence<sizeof...(Args)>{});
        return ans;
    }
};

template <typename T>
__host__ __device__ void divide_fn(T &x, const int &y) {
    x /= static_cast<float>(y);
}

template <typename T, std::size_t... Is>
__host__ __device__ void divide_tuple_impl(T &t,
                                           const int &y,
                                           thrust::integer_sequence<std::size_t, Is...>) {
    std::initializer_list<int>{((void)divide_fn(thrust::get<Is>(t), y), 0)...};
}

template <class... Args>
struct divide_tuple_functor
    : public thrust::binary_function<const thrust::tuple<Args...>,
                                     const int,
                                     thrust::tuple<Args...>> {
    __host__ __device__ thrust::tuple<Args...> operator()(
            const thrust::tuple<Args...> &x, const int &y) const {
        thrust::tuple<Args...> ans = x;
        divide_tuple_impl(ans, y,
                          thrust::make_index_sequence<sizeof...(Args)>{});
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

template <class... Args>
auto enumerate_iterator(size_t n, Args &... args) {
    return make_tuple_iterator(thrust::make_counting_iterator<size_t>(n),
                               args...);
}

template <class... Args>
auto enumerate_iterator(size_t n, const Args &... args) {
    return make_tuple_iterator(thrust::make_counting_iterator<size_t>(n),
                               args...);
}

template <class... Args>
auto enumerate_begin(Args &... args) {
    return make_tuple_iterator(thrust::make_counting_iterator<size_t>(0),
                               std::begin(args)...);
}

template <class... Args>
auto enumerate_begin(const Args &... args) {
    return make_tuple_iterator(thrust::make_counting_iterator<size_t>(0),
                               std::begin(args)...);
}

template <class T, class... Args>
auto enumerate_end(T &first, Args &... args) {
    return make_tuple_iterator(thrust::make_counting_iterator(first.size()),
                               std::end(first), std::end(args)...);
}

template <class T, class... Args>
auto enumerate_end(const T &first, const Args &... args) {
    return make_tuple_iterator(thrust::make_counting_iterator(first.size()),
                               std::end(first), std::end(args)...);
}

template <class T>
void resize_fn(size_t new_size, T &a) {
    a.resize(new_size);
}

template <class... Args>
void resize_all(size_t new_size, Args &... args) {
    std::initializer_list<int>{((void)resize_fn(new_size, args), 0)...};
}

template <typename DerivedPolicy, class Func, class... Args>
size_t remove_if_vectors_without_resize(
        const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        Func fn,
        utility::device_vector<Args> &... args) {
    auto begin = make_tuple_begin(args...);
    auto end = thrust::remove_if(exec, begin, make_tuple_end(args...), fn);
    return thrust::distance(begin, end);
}

template <typename DerivedPolicy, class Func, class... Args>
size_t remove_if_vectors(
        const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        Func fn,
        utility::device_vector<Args> &... args) {
    size_t k = remove_if_vectors_without_resize(exec, fn, args...);
    resize_all(k, args...);
    return k;
}

template <class Func, class... Args>
size_t remove_if_vectors_without_resize(
        Func fn, utility::device_vector<Args> &... args) {
    return remove_if_vectors_without_resize(thrust::device, fn, args...);
}

template <class Func, class... Args>
size_t remove_if_vectors(Func fn, utility::device_vector<Args> &... args) {
    return remove_if_vectors(thrust::device, fn, args...);
}

template <typename T>
struct swap_index_functor {
    __device__ Eigen::Matrix<T, 2, 1> operator()(
            const Eigen::Matrix<T, 2, 1> &x) {
        return Eigen::Matrix<T, 2, 1>(x[1], x[0]);
    };
};

template <typename T>
void swap_index(utility::device_vector<Eigen::Matrix<T, 2, 1>> &v) {
    thrust::transform(v.begin(), v.end(), v.begin(), swap_index_functor<T>());
}

template <int Dim>
void remove_negative(utility::device_vector<Eigen::Matrix<int, Dim, 1>> &idxs) {
    remove_negative(thrust::device, idxs);
}

template <typename T>
void remove_scalar_negative(utility::device_vector<T> &idxs) {
    remove_scalar_negative(thrust::device, idxs);
}

template <typename DerivedPolicy, int Dim>
void remove_negative(
        const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        utility::device_vector<Eigen::Matrix<int, Dim, 1>> &idxs) {
    auto end = thrust::remove_if(
            exec, idxs.begin(), idxs.end(),
            [] __device__(const Eigen::Matrix<int, Dim, 1> &idx) {
                return Eigen::device_any(idx.array() < 0);
            });
    idxs.resize(thrust::distance(idxs.begin(), end));
}

template <typename DerivedPolicy, typename T>
void remove_scalar_negative(
        const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        utility::device_vector<T> &idxs) {
    auto end =
            thrust::remove_if(exec, idxs.begin(), idxs.end(),
                              [] __device__(const T &idx) { return idx < 0; });
    idxs.resize(thrust::distance(idxs.begin(), end));
}

template <typename T, int Index>
struct element_get_functor {
    __device__ typename T::Scalar operator()(const T &x) {
        return x[Index];
    };
};

template <int Index, typename T, class... Args>
struct tuple_get_functor {
    __device__ T operator()(const thrust::tuple<Args...> &x) {
        return thrust::get<Index>(x);
    };
};

__host__ __device__ inline int IndexOf(int x, int y, int z, int resolution) {
    return x * resolution * resolution + y * resolution + z;
}

__host__ __device__ inline int IndexOf(const Eigen::Vector3i &xyz,
                                       int resolution) {
    return IndexOf(xyz(0), xyz(1), xyz(2), resolution);
}

__host__ __device__ inline thrust::tuple<int, int, int> KeyOf(size_t idx,
                                                              int resolution) {
    int res2 = resolution * resolution;
    int x = idx / res2;
    int yz = idx % res2;
    int y = yz / resolution;
    int z = yz % resolution;
    return thrust::make_tuple(x, y, z);
}

template <typename T, typename ContainerType>
inline void copy_device_to_host(const utility::device_vector<T> &src,
                                ContainerType &dist) {
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(dist.data()),
                            thrust::raw_pointer_cast(src.data()),
                            src.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, typename ContainerType>
inline void copy_host_to_device(const ContainerType &src,
                                utility::device_vector<T> &dist) {
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(dist.data()),
                            thrust::raw_pointer_cast(src.data()),
                            src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T, typename ContainerType>
inline void copy_device_to_device(const utility::device_vector<T> &src,
                                  ContainerType &dist) {
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(dist.data()),
                            thrust::raw_pointer_cast(src.data()),
                            src.size() * sizeof(T), cudaMemcpyDeviceToDevice));
}

namespace utility {

template <typename T>
struct hash_eigen {
    __host__ __device__ std::size_t operator()(T const &matrix) const {
        size_t seed = 0;
#pragma unroll
        for (int i = 0; i < T::SizeAtCompileTime; i++) {
            auto elem = *(matrix.data() + i);
            seed ^= stdgpu::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                    (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

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

}  // namespace utility
}  // namespace cupoch