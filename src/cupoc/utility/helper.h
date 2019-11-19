#pragma once
#include <Eigen/Core>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

namespace thrust {

template<>
struct equal_to<Eigen::Vector3i> {
    typedef Eigen::Vector3i first_argument_type;
    typedef Eigen::Vector3i second_argument_type;
    typedef bool result_type;
    __thrust_exec_check_disable__
    __host__ __device__ bool operator()(const Eigen::Vector3i &lhs, const Eigen::Vector3i &rhs) const {
        return (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2]);
    }
};

template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type, difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

}

namespace cupoc {

template<class... Args>
struct add_tuple_functor : public thrust::binary_function<const thrust::tuple<Args...>, const thrust::tuple<Args...>, thrust::tuple<Args...>> {
    __host__ __device__
    thrust::tuple<Args...> operator()(const thrust::tuple<Args...>& x, const thrust::tuple<Args...>& y) const;
};

template<class... Args>
struct devided_tuple_functor : public thrust::binary_function<const thrust::tuple<Args...>, const int, thrust::tuple<Args...>> {
    __host__ __device__
    thrust::tuple<Args...> operator()(const thrust::tuple<Args...>& x, const int& y) const;
};

template<class T1>
struct add_tuple_functor<T1> : public thrust::binary_function<const thrust::tuple<T1>, const thrust::tuple<T1>, thrust::tuple<T1>> {
    __host__ __device__
    thrust::tuple<T1> operator()(const thrust::tuple<T1>& x, const thrust::tuple<T1>& y) const {
        thrust::tuple<T1> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) + thrust::get<0>(y);
        return ans;
    }
};

template<class T1, class T2>
struct add_tuple_functor<T1, T2> : public thrust::binary_function<const thrust::tuple<T1, T2>, const thrust::tuple<T1, T2>, thrust::tuple<T1, T2>> {
    __host__ __device__
    thrust::tuple<T1, T2> operator()(const thrust::tuple<T1, T2>& x, const thrust::tuple<T1, T2>& y) const {
        thrust::tuple<T1, T2> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) + thrust::get<0>(y);
        thrust::get<1>(ans) = thrust::get<1>(x) + thrust::get<1>(y);
        return ans;
    }
};

template<class T1, class T2, class T3>
struct add_tuple_functor<T1, T2, T3> : public thrust::binary_function<const thrust::tuple<T1, T2, T3>, const thrust::tuple<T1, T2, T3>, thrust::tuple<T1, T2, T3>> {
    __host__ __device__
    thrust::tuple<T1, T2, T3> operator()(const thrust::tuple<T1, T2, T3>& x, const thrust::tuple<T1, T2, T3>& y) const {
        thrust::tuple<T1, T2, T3> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) + thrust::get<0>(y);
        thrust::get<1>(ans) = thrust::get<1>(x) + thrust::get<1>(y);
        thrust::get<2>(ans) = thrust::get<2>(x) + thrust::get<2>(y);
        return ans;
    }
};

template<class T1>
struct devided_tuple_functor<T1> : public thrust::binary_function<const thrust::tuple<T1>, const int, thrust::tuple<T1>> {
    __host__ __device__
    thrust::tuple<T1> operator()(const thrust::tuple<T1>& x, const int& y) const {
        thrust::tuple<T1> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) / static_cast<float>(y);
        return ans;
    }
};

template<class T1, class T2>
struct devided_tuple_functor<T1, T2> : public thrust::binary_function<const thrust::tuple<T1, T2>, const int, thrust::tuple<T1, T2>> {
    __host__ __device__
    thrust::tuple<T1, T2> operator()(const thrust::tuple<T1, T2>& x, const int& y) const {
        thrust::tuple<T1, T2> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) / static_cast<float>(y);
        thrust::get<1>(ans) = thrust::get<1>(x) / static_cast<float>(y);
        return ans;
    }
};

template<class T1, class T2, class T3>
struct devided_tuple_functor<T1, T2, T3> : public thrust::binary_function<const thrust::tuple<T1, T2, T3>, const int, thrust::tuple<T1, T2, T3>> {
    __host__ __device__
    thrust::tuple<T1, T2, T3> operator()(const thrust::tuple<T1, T2, T3>& x, const int& y) const {
        thrust::tuple<T1, T2, T3> ans;
        thrust::get<0>(ans) = thrust::get<0>(x) / static_cast<float>(y);
        thrust::get<1>(ans) = thrust::get<1>(x) / static_cast<float>(y);
        thrust::get<2>(ans) = thrust::get<2>(x) / static_cast<float>(y);
        return ans;
    }
};

}