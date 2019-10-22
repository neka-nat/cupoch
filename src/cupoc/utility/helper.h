#pragma once
#include <Eigen/Core>
#include <thrust/functional.h>

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

}