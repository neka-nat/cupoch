#include "cupoch_pybind/device_vector_wrapper.h"
#include "cupoch/geometry/pointcloud.h"

namespace cupoch {
namespace wrapper {

template<typename Type>
device_vector_wrapper<Type>::device_vector_wrapper() {};
template<typename Type>
device_vector_wrapper<Type>::device_vector_wrapper(const device_vector_wrapper<Type>& other) {data_ = other.data_;}
template<typename Type>
device_vector_wrapper<Type>::device_vector_wrapper(const thrust::host_vector<Type>& other) {data_ = other;}
template<typename Type>
device_vector_wrapper<Type>::device_vector_wrapper(const utility::device_vector<Type>& other) {data_ = other;}
template<typename Type>
device_vector_wrapper<Type>::~device_vector_wrapper() {};

template<typename Type>
device_vector_wrapper<Type> &device_vector_wrapper<Type>::operator=(const device_vector_wrapper<Type> &other) {
    data_ = other.data_;
    return *this;
}

template<typename Type>
thrust::host_vector<Type> device_vector_wrapper<Type>::cpu() const {
    thrust::host_vector<Type> ans = data_;
    return ans;
}

template class device_vector_wrapper<Eigen::Vector3f>;
template class device_vector_wrapper<Eigen::Vector2f>;
template class device_vector_wrapper<Eigen::Vector3i>;
template class device_vector_wrapper<Eigen::Vector2i>;
template class device_vector_wrapper<float>;
template class device_vector_wrapper<int>;
template class device_vector_wrapper<size_t>;

template<typename Type>
void FromWrapper(utility::device_vector<Type>& dv, const device_vector_wrapper<Type>& vec) {
    dv = vec.data_;
}

template void FromWrapper<Eigen::Vector3f>(utility::device_vector<Eigen::Vector3f>& dv, const device_vector_wrapper<Eigen::Vector3f>& vec);
template void FromWrapper<Eigen::Vector2f>(utility::device_vector<Eigen::Vector2f>& dv, const device_vector_wrapper<Eigen::Vector2f>& vec);
template void FromWrapper<Eigen::Vector3i>(utility::device_vector<Eigen::Vector3i>& dv, const device_vector_wrapper<Eigen::Vector3i>& vec);
template void FromWrapper<Eigen::Vector2i>(utility::device_vector<Eigen::Vector2i>& dv, const device_vector_wrapper<Eigen::Vector2i>& vec);
template void FromWrapper<float>(utility::device_vector<float>& dv, const device_vector_wrapper<float>& vec);
template void FromWrapper<int>(utility::device_vector<int>& dv, const device_vector_wrapper<int>& vec);
template void FromWrapper<size_t>(utility::device_vector<size_t>& dv, const device_vector_wrapper<size_t>& vec);

}
}