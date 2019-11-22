#include "cupoc/registration/transformation_estimation.h"
#include "cupoc/registration/kabsch.h"
#include "cupoc/geometry/pointcloud.h"

using namespace cupoc;
using namespace cupoc::registration;

namespace{

template<int Index>
struct element_copy_functor {
    element_copy_functor(const Eigen::Vector3f_u* points) : points_(points) {};
    const Eigen::Vector3f_u* points_;
    __device__
    Eigen::Vector3f_u operator()(const thrust::tuple<int, int>& x) const {
        return points_[thrust::get<Index>(x)];
    }
};

struct diff_square_functor {
    __device__
    float operator()(const Eigen::Vector3f_u& x, const Eigen::Vector3f_u& y) const {
        return (x - y).squaredNorm();
    }
};

}

float TransformationEstimationPointToPoint::ComputeRMSE(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const CorrespondenceSet &corres) const {
    thrust::device_vector<float> sqr_errs(corres.size());
    thrust::device_vector<Eigen::Vector3f_u> src_cor(corres.size());
    thrust::device_vector<Eigen::Vector3f_u> tgt_cor(corres.size());
    thrust::transform(corres.begin(), corres.end(), src_cor.begin(),
                      element_copy_functor<0>(thrust::raw_pointer_cast(source.points_.data())));
    thrust::transform(corres.begin(), corres.end(), tgt_cor.begin(),
                      element_copy_functor<1>(thrust::raw_pointer_cast(target.points_.data())));
    diff_square_functor func;
    thrust::transform(src_cor.begin(), src_cor.end(), tgt_cor.begin(), sqr_errs.begin(), func);
    const float err = thrust::reduce(sqr_errs.begin(), sqr_errs.end());
    return std::sqrt(err / (float)corres.size());
}

Eigen::Matrix4f TransformationEstimationPointToPoint::ComputeTransformation(
    const geometry::PointCloud &source,
    const geometry::PointCloud &target,
    const CorrespondenceSet &corres) const {
    thrust::device_vector<Eigen::Vector3f_u> src_cor(corres.size());
    thrust::device_vector<Eigen::Vector3f_u> tgt_cor(corres.size());
    thrust::transform(corres.begin(), corres.end(), src_cor.begin(),
                      element_copy_functor<0>(thrust::raw_pointer_cast(source.points_.data())));
    thrust::transform(corres.begin(), corres.end(), tgt_cor.begin(),
                      element_copy_functor<1>(thrust::raw_pointer_cast(target.points_.data())));
    return Kabsch(src_cor, tgt_cor);
}
