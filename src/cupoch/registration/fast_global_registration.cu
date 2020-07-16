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
#include <thrust/random.h>

#include "cupoch/geometry/bruteforce_nn.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/registration/fast_global_registration.h"
#include "cupoch/registration/registration.h"

namespace cupoch {
namespace registration {
namespace {

struct compute_tuple_constraint_functor {
    compute_tuple_constraint_functor(
            size_t ncorr,
            const thrust::tuple<int, int>* corres_cross,
            const Eigen::Vector3f* point_cloud_vec_fi_points,
            const Eigen::Vector3f* point_cloud_vec_fj_points,
            thrust::tuple<int, int>* corres_tuple,
            float scale)
        : ncorr_(ncorr),
          corres_cross_(corres_cross),
          point_cloud_vec_fi_points_(point_cloud_vec_fi_points),
          point_cloud_vec_fj_points_(point_cloud_vec_fj_points),
          corres_tuple_(corres_tuple),
          scale_(scale){};
    const int ncorr_;
    const thrust::tuple<int, int>* corres_cross_;
    const Eigen::Vector3f* point_cloud_vec_fi_points_;
    const Eigen::Vector3f* point_cloud_vec_fj_points_;
    thrust::tuple<int, int>* corres_tuple_;
    const float scale_;
    __device__ void operator()(size_t idx) {
        int rand0, rand1, rand2;
        int idi0, idi1, idi2, idj0, idj1, idj2;
        thrust::default_random_engine eng;
        thrust::uniform_int_distribution<int> dist(0, ncorr_ - 1);
        eng.discard(idx);
        rand0 = dist(eng);
        rand1 = dist(eng);
        rand2 = dist(eng);
        idi0 = thrust::get<0>(corres_cross_[rand0]);
        idj0 = thrust::get<1>(corres_cross_[rand0]);
        idi1 = thrust::get<0>(corres_cross_[rand1]);
        idj1 = thrust::get<1>(corres_cross_[rand1]);
        idi2 = thrust::get<0>(corres_cross_[rand2]);
        idj2 = thrust::get<1>(corres_cross_[rand2]);

        // collect 3 points from i-th fragment
        Eigen::Vector3f pti0 = point_cloud_vec_fi_points_[idi0];
        Eigen::Vector3f pti1 = point_cloud_vec_fi_points_[idi1];
        Eigen::Vector3f pti2 = point_cloud_vec_fi_points_[idi2];
        float li0 = (pti0 - pti1).norm();
        float li1 = (pti1 - pti2).norm();
        float li2 = (pti2 - pti0).norm();

        // collect 3 points from j-th fragment
        Eigen::Vector3f ptj0 = point_cloud_vec_fj_points_[idj0];
        Eigen::Vector3f ptj1 = point_cloud_vec_fj_points_[idj1];
        Eigen::Vector3f ptj2 = point_cloud_vec_fj_points_[idj2];
        float lj0 = (ptj0 - ptj1).norm();
        float lj1 = (ptj1 - ptj2).norm();
        float lj2 = (ptj2 - ptj0).norm();

        // check tuple constraint
        bool cond = (li0 * scale_ < lj0) && (lj0 < li0 / scale_) &&
                    (li1 * scale_ < lj1) && (lj1 < li1 / scale_) &&
                    (li2 * scale_ < lj2) && (lj2 < li2 / scale_);
        thrust::tuple<int, int> invalid_idx = thrust::make_tuple(-1, -1);
        corres_tuple_[3 * idx] =
                (cond) ? thrust::make_tuple(idi0, idj0) : invalid_idx;
        corres_tuple_[3 * idx + 1] =
                (cond) ? thrust::make_tuple(idi1, idj1) : invalid_idx;
        corres_tuple_[3 * idx + 2] =
                (cond) ? thrust::make_tuple(idi2, idj2) : invalid_idx;
    }
};

template <int Dim>
utility::device_vector<thrust::tuple<int, int>> AdvancedMatching(
        const std::vector<geometry::PointCloud>& point_cloud_vec,
        const std::vector<Feature<Dim>>& features_vec,
        const FastGlobalRegistrationOption& option) {
    // STEP 0) Swap source and target if necessary
    int fi = 0, fj = 1;
    utility::LogDebug("Advanced matching : [{:d} - {:d}]", fi, fj);
    bool swapped = false;
    if (point_cloud_vec[fj].points_.size() >
        point_cloud_vec[fi].points_.size()) {
        int temp = fi;
        fi = fj;
        fj = temp;
        swapped = true;
    }

    // STEP 1) Initial matching
    int nPti = int(point_cloud_vec[fi].points_.size());
    int nPtj = int(point_cloud_vec[fj].points_.size());
    utility::device_vector<int> corresK;
    utility::device_vector<float> dis;
    utility::device_vector<thrust::tuple<int, int>> corres;
    corres.resize(nPti + nPtj);
    geometry::BruteForceNN<Dim>(features_vec[fi].data_, features_vec[fj].data_,
                                corresK, dis);
    thrust::copy(make_tuple_iterator(corresK.begin(),
                                     thrust::make_counting_iterator<int>(0)),
                 make_tuple_iterator(
                         corresK.end(),
                         thrust::make_counting_iterator<int>(corresK.size())),
                 corres.begin());
    geometry::BruteForceNN<Dim>(features_vec[fj].data_, features_vec[fi].data_,
                                corresK, dis);
    thrust::copy(make_tuple_iterator(thrust::make_counting_iterator<int>(0),
                                     corresK.begin()),
                 make_tuple_iterator(
                         thrust::make_counting_iterator<int>(corresK.size()),
                         corresK.end()),
                 corres.begin() + nPtj);
    thrust::sort(corres.begin(), corres.end());
    utility::LogDebug("points are remained : {:d}", corres.size());

    // STEP 2) CROSS CHECK
    utility::LogDebug("\t[cross check] ");
    utility::device_vector<thrust::tuple<int, int>> corres_cross(corres.size());
    utility::device_vector<int> counts(corres.size());
    auto end1 = thrust::reduce_by_key(corres.begin(), corres.end(),
                                      thrust::make_constant_iterator<int>(1),
                                      corres_cross.begin(), counts.begin());
    auto end2 =
            thrust::remove_if(corres_cross.begin(), end1.first, counts.begin(),
                              [] __device__(int cnt) { return cnt < 2; });
    corres_cross.resize(thrust::distance(corres_cross.begin(), end2));
    utility::LogDebug("points are remained : {:d}", corres_cross.size());

    // STEP 3) TUPLE CONSTRAINT
    utility::LogDebug("\t[tuple constraint] ");
    float scale = option.tuple_scale_;
    size_t ncorr = corres_cross.size();
    size_t number_of_trial = ncorr * 100;

    utility::device_vector<thrust::tuple<int, int>> corres_tuple(
            3 * number_of_trial);
    compute_tuple_constraint_functor func(
            ncorr, thrust::raw_pointer_cast(corres_cross.data()),
            thrust::raw_pointer_cast(point_cloud_vec[fi].points_.data()),
            thrust::raw_pointer_cast(point_cloud_vec[fj].points_.data()),
            thrust::raw_pointer_cast(corres_tuple.data()), scale);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator(number_of_trial), func);
    auto end3 = thrust::remove_if(
            corres_tuple.begin(), corres_tuple.end(),
            [] __device__(const thrust::tuple<int, int>& corr) {
                return thrust::get<0>(corr) < 0;
            });
    size_t n_res = thrust::distance(corres_tuple.begin(), end3);
    corres_tuple.resize(std::min((int)n_res, option.maximum_tuple_count_));
    utility::LogDebug("{:d} tuples ({:d} trial, {:d} actual).",
                      corres_tuple.size(), number_of_trial, n_res);

    if (swapped) {
        thrust::for_each(corres_tuple.begin(), corres_tuple.end(),
                         [] __device__(thrust::tuple<int, int> & corr) {
                             thrust::swap(thrust::get<0>(corr),
                                          thrust::get<1>(corr));
                         });
    }
    utility::LogDebug("\t[final] matches {:d}.", (int)corres_tuple.size());
    return corres_tuple;
}

// Normalize scale of points. X' = (X-\mu)/scale
std::tuple<std::vector<Eigen::Vector3f>, float, float> NormalizePointCloud(
        std::vector<geometry::PointCloud>& point_cloud_vec,
        const FastGlobalRegistrationOption& option) {
    int num = 2;
    float scale = 0;
    std::vector<Eigen::Vector3f> pcd_mean_vec;
    float scale_global, scale_start;

    for (int i = 0; i < num; ++i) {
        Eigen::Vector3f mean =
                thrust::reduce(point_cloud_vec[i].points_.begin(),
                               point_cloud_vec[i].points_.end(),
                               Eigen::Vector3f(0.0, 0.0, 0.0),
                               thrust::plus<Eigen::Vector3f>());
        mean = mean / point_cloud_vec[i].points_.size();
        pcd_mean_vec.push_back(mean);

        utility::LogDebug("normalize points :: mean = [{:f} {:f} {:f}]",
                          mean(0), mean(1), mean(2));
        thrust::for_each(
                point_cloud_vec[i].points_.begin(),
                point_cloud_vec[i].points_.end(),
                [mean] __device__(Eigen::Vector3f & pt) { pt -= mean; });

        scale = thrust::transform_reduce(
                point_cloud_vec[i].points_.begin(),
                point_cloud_vec[i].points_.end(),
                [] __device__(const Eigen::Vector3f& pt) { return pt.norm(); },
                scale, thrust::maximum<float>());
    }

    if (option.use_absolute_scale_) {
        scale_global = 1.0;
        scale_start = scale;
    } else {
        scale_global = scale;
        scale_start = 1.0;
    }
    utility::LogDebug("normalize points :: global scale : {:f}", scale_global);

    for (int i = 0; i < num; ++i) {
        thrust::for_each(point_cloud_vec[i].points_.begin(),
                         point_cloud_vec[i].points_.end(),
                         [scale_global] __device__(Eigen::Vector3f & pt) {
                             pt /= scale_global;
                         });
    }
    return std::make_tuple(pcd_mean_vec, scale_global, scale_start);
}

struct compute_jacobian_functor {
    compute_jacobian_functor(float par) : par_(par){};
    const float par_;
    __device__ thrust::tuple<Eigen::Matrix6f, Eigen::Vector6f> operator()(
            const thrust::tuple<Eigen::Vector3f, Eigen::Vector3f>& x) const {
        Eigen::Vector3f p, q;
        p = thrust::get<0>(x);
        q = thrust::get<1>(x);
        Eigen::Vector3f rpq = p - q;
        float temp = par_ / (rpq.dot(rpq) + par_);
        float s = temp * temp;
        float r = 0;

        Eigen::Matrix6f JTJ = Eigen::Matrix6f::Zero();
        Eigen::Vector6f JTr = Eigen::Vector6f::Zero();
        Eigen::Vector6f J = Eigen::Vector6f::Zero();
        J(1) = -q(2);
        J(2) = q(1);
        J(3) = -1;
        r = rpq(0);
        JTJ += J * J.transpose() * s;
        JTr += J * r * s;

        J.setZero();
        J(2) = -q(0);
        J(0) = q(2);
        J(4) = -1;
        r = rpq(1);
        JTJ += J * J.transpose() * s;
        JTr += J * r * s;

        J.setZero();
        J(0) = -q(1);
        J(1) = q(0);
        J(5) = -1;
        r = rpq(2);
        JTJ += J * J.transpose() * s;
        JTr += J * r * s;
        return thrust::make_tuple(JTJ, JTr);
    }
};

Eigen::Matrix4f OptimizePairwiseRegistration(
        const std::vector<geometry::PointCloud>& point_cloud_vec,
        const utility::device_vector<thrust::tuple<int, int>>& corres,
        float scale_start,
        const FastGlobalRegistrationOption& option) {
    utility::LogDebug("Pairwise rigid pose optimization");
    float par = scale_start;
    int numIter = option.iteration_number_;

    int i = 0, j = 1;
    geometry::PointCloud point_cloud_copy_j = point_cloud_vec[j];

    if (corres.size() < 10) return Eigen::Matrix4f::Identity();
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();

    for (int itr = 0; itr < numIter; itr++) {
        Eigen::Matrix6f JTJ = Eigen::Matrix6f::Zero();
        Eigen::Vector6f JTr = Eigen::Vector6f::Zero();
        compute_jacobian_functor func(par);
        thrust::tie(JTJ, JTr) = thrust::transform_reduce(
                make_tuple_iterator(
                        thrust::make_permutation_iterator(
                                point_cloud_vec[i].points_.begin(),
                                thrust::make_transform_iterator(
                                        corres.begin(),
                                        tuple_get_functor<0, int, int, int>())),
                        thrust::make_permutation_iterator(
                                point_cloud_copy_j.points_.begin(),
                                thrust::make_transform_iterator(
                                        corres.begin(),
                                        tuple_get_functor<1, int, int,
                                                          int>()))),
                make_tuple_iterator(
                        thrust::make_permutation_iterator(
                                point_cloud_vec[i].points_.begin(),
                                thrust::make_transform_iterator(
                                        corres.end(),
                                        tuple_get_functor<0, int, int, int>())),
                        thrust::make_permutation_iterator(
                                point_cloud_copy_j.points_.begin(),
                                thrust::make_transform_iterator(
                                        corres.end(),
                                        tuple_get_functor<1, int, int,
                                                          int>()))),
                func, thrust::make_tuple(JTJ, JTr),
                add_tuple_functor<Eigen::Matrix6f, Eigen::Vector6f>());
        bool success;
        Eigen::Vector6f result;
        thrust::tie(success, result) =
                utility::SolveLinearSystemPSD<6>(-JTJ, JTr);
        Eigen::Matrix4f delta = utility::TransformVector6fToMatrix4f(result);
        trans = delta * trans;
        point_cloud_copy_j.Transform(delta);

        // graduated non-convexity.
        if (option.decrease_mu_) {
            if (itr % 4 == 0 && par > option.maximum_correspondence_distance_) {
                par /= option.division_factor_;
            }
        }
    }
    return trans;
}

// Below line indicates how the transformation matrix aligns two point clouds
// e.g. T * point_cloud_vec[1] is aligned with point_cloud_vec[0].
Eigen::Matrix4f GetInvTransformationOriginalScale(
        const Eigen::Matrix4f& transformation,
        const std::vector<Eigen::Vector3f>& pcd_mean_vec,
        float scale_global) {
    Eigen::Matrix3f R = transformation.block<3, 3>(0, 0);
    Eigen::Vector3f t = transformation.block<3, 1>(0, 3);
    Eigen::Matrix4f transtemp = Eigen::Matrix4f::Zero();
    transtemp.block<3, 3>(0, 0) = R.transpose();
    transtemp.block<3, 1>(0, 3) =
            -R.transpose() *
            (-R * pcd_mean_vec[1] + t * scale_global + pcd_mean_vec[0]);
    transtemp(3, 3) = 1;
    return transtemp;
}

}  // namespace

template<int Dim>
RegistrationResult FastGlobalRegistration(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const Feature<Dim>& source_feature,
        const Feature<Dim>& target_feature,
        const FastGlobalRegistrationOption& option /* =
        FastGlobalRegistrationOption()*/) {
    std::vector<geometry::PointCloud> point_cloud_vec;
    geometry::PointCloud source_orig = source;
    geometry::PointCloud target_orig = target;
    point_cloud_vec.push_back(source);
    point_cloud_vec.push_back(target);

    std::vector<Feature<Dim>> features_vec;
    features_vec.push_back(source_feature);
    features_vec.push_back(target_feature);

    float scale_global, scale_start;
    std::vector<Eigen::Vector3f> pcd_mean_vec;
    std::tie(pcd_mean_vec, scale_global, scale_start) =
            NormalizePointCloud(point_cloud_vec, option);
    utility::device_vector<thrust::tuple<int, int>> corres;
    corres = AdvancedMatching<Dim>(point_cloud_vec, features_vec, option);
    Eigen::Matrix4f transformation;
    transformation = OptimizePairwiseRegistration(point_cloud_vec, corres,
                                                  scale_global, option);

    // as the original code T * point_cloud_vec[1] is aligned with
    // point_cloud_vec[0] matrix inverse is applied here.
    return EvaluateRegistration(
            source_orig, target_orig, option.maximum_correspondence_distance_,
            GetInvTransformationOriginalScale(transformation, pcd_mean_vec,
                                              scale_global));
}

template RegistrationResult FastGlobalRegistration<33>(
        const geometry::PointCloud& source,
        const geometry::PointCloud& target,
        const Feature<33>& source_feature,
        const Feature<33>& target_feature,
        const FastGlobalRegistrationOption& option);

}  // namespace registration
}  // namespace cupoch