/**
 * Copyright (c) 2021 Neka-Nat
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
#include <Eigen/Geometry>

#include "cupoch/geometry/kdtree_flann.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/registration/feature.h"
#include "cupoch/utility/eigenvalue.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace registration {

__constant__ float PST_RAD_45 = 0.78539816339744830961566084581988;
__constant__ float PST_RAD_90 = 1.5707963267948966192313216916398;
__constant__ float PST_RAD_135 = 2.3561944901923449288469825374596;
__constant__ float PST_RAD_PI_7_8 = 2.7488935718910690836548129603691;

namespace {
    struct compute_shot_functor {
        compute_shot_functor(const Eigen::Vector3f *points,
                             const Eigen::Vector3f *normals,
                             const int *indices,
                             const float *distance2,
                             float radius,
                             int knn)
        : points_(points), normals_(normals),
        indices_(indices), distance2_(distance2),
        radius_(radius), radius1_2_(radius * 0.5),
        radius3_4_(radius * 3.0 / 4.0), radius1_4_(radius * 0.25),
        knn_(knn) {};
        const Eigen::Vector3f *points_;
        const Eigen::Vector3f *normals_;
        const int *indices_;
        const float *distance2_;
        const float radius_;
        const float radius1_2_;
        const float radius3_4_;
        const float radius1_4_;
        const int knn_;
        const int n_bins_ = 10;
        const int min_neighbors_ = 5;
        const int max_angular_sectors_ = 32;
        __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, int> compute_shot_lrf(size_t idx, const Eigen::Vector3f& point_i) const {
            Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
            float w_total = 0.0;
            int n_nb = 0;
            for (size_t k = 0; k < knn_; k++) {
                int idx_knn = indices_[idx * knn_ + k];
                if (idx_knn < 0 || idx == idx_knn) continue;
                const Eigen::Vector3f q = points_[idx_knn] - point_i;
                const float dist2 = distance2_[idx * knn_ + k];
                const float w = radius_ - sqrt(dist2);
                cov += w * q * q.transpose();
                w_total += w;
                n_nb += 1;
            }
            cov /= w_total;
            auto evecs = utility::FastEigen3x3MinMaxVec(cov);
            Eigen::Vector3f zaxis = thrust::get<0>(evecs);
            Eigen::Vector3f xaxis = thrust::get<1>(evecs);
            int n_px = 0;
            int n_pz = 0;
            for (size_t k = 0; k < knn_; k++) {
                int idx_knn = indices_[idx * knn_ + k];
                if (idx_knn < 0 || idx == idx_knn) continue;
                const Eigen::Vector3f q = points_[idx_knn] - point_i;
                n_px += int(q.dot(xaxis) >= 0);
                n_pz += int(q.dot(zaxis) >= 0);
            }
            if (n_px < n_nb - n_px) {
                xaxis *= -1.0;
            }
            if (n_pz < n_nb - n_pz) {
                zaxis *= -1.0;
            }
            Eigen::Vector3f yaxis = zaxis.cross(xaxis);
            return thrust::make_tuple(xaxis, yaxis, zaxis, n_nb);
        }

        __device__ Feature<352>::FeatureType operator()(size_t idx) const {
            Feature<352>::FeatureType ft = Feature<352>::FeatureType::Zero();
            const Eigen::Vector3f point_i = points_[idx];
            const Eigen::Vector3f normal_i = normals_[idx];
            auto lrf = compute_shot_lrf(idx, point_i);
            const int n_nb = thrust::get<3>(lrf);
            if (n_nb < min_neighbors_) {
                return ft;
            }
            for (size_t k = 0; k < knn_; k++) {
                int idx_knn = indices_[idx * knn_ + k];
                if (idx_knn < 0 || idx == idx_knn) continue;
                const Eigen::Vector3f q = points_[idx_knn] - point_i;
                const float dist = sqrt(distance2_[idx * knn_ + k]);
                if (dist == 0) continue;
                const float cos_desc = min(max(thrust::get<2>(lrf).dot(normal_i), -1.0), 1.0);
                float bindist = ((1.0 + cos_desc) * n_bins_) / 2.0;
    
                float x_lrf = q.dot(thrust::get<0>(lrf));
                float y_lrf = q.dot(thrust::get<1>(lrf));
                float z_lrf = q.dot(thrust::get<2>(lrf));
                if (abs(x_lrf) < 1.0e-30) {
                    x_lrf = 0.0;
                }
                if (abs(y_lrf) < 1.0e-30) {
                    y_lrf = 0.0;
                }
                if (abs(z_lrf) < 1.0e-30) {
                    z_lrf = 0.0;
                }
                unsigned char bit4 = (y_lrf > 0 || (y_lrf == 0.0 && x_lrf < 0)) ? 1 : 0;
                unsigned char bit3 = (unsigned char) ((x_lrf > 0 || (x_lrf == 0.0 && y_lrf > 0)) ? !bit4 : bit4);
                int desc_index = (bit4 << 3) + (bit3 << 2);
                desc_index = desc_index << 1;
                if (x_lrf * y_lrf > 0 || x_lrf == 0.0) {
                    desc_index += (abs(x_lrf) >= abs(y_lrf)) ? 0 : 4;
                } else {
                    desc_index += (abs(x_lrf) > abs(y_lrf)) ? 4 : 0;
                }
                desc_index += z_lrf > 0 ? 1 : 0;
                // 2 RADII
                desc_index += (dist > radius1_2_) ? 2 : 0;
                int step_index = (bindist < 0.0) ? (int)(ceilf(bindist - 0.5)) : (int)(floorf(bindist + 0.5));
                int volume_index = desc_index * (n_bins_ + 1);
                //Interpolation on the cosine (adjacent bins in the histogram)
                bindist -= step_index;
                float init_weight = 1 - abs(bindist);
                if (bindist > 0) {
                    ft[volume_index + ((step_index + 1) % n_bins_)] += bindist;
                } else {
                    ft[volume_index + ((step_index - 1 + n_bins_) % n_bins_)] += -bindist;
                }
                //Interpolation on the distance (adjacent husks)
                if (dist > radius1_2_) {
                    float radius_dist = (dist - radius3_4_) / radius1_2_;
                    if (dist > radius3_4_){
                        init_weight += 1 - radius_dist;
                    } else {
                        init_weight += 1 + radius_dist;
                        ft[(desc_index - 2) * (n_bins_ + 1) + step_index] -= radius_dist;
                    }
                } else {
                    float radius_dist = (dist - radius1_4_) / radius1_2_;
                    if (dist < radius1_4_) {
                        init_weight += 1 + radius_dist;
                    } else {
                        init_weight += 1 - radius_dist;
                        ft[(desc_index + 2) * (n_bins_ + 1) + step_index] += radius_dist;
                    }
                }

                //Interpolation on the inclination (adjacent vertical volumes)
                float inclination_cos = min(max(z_lrf / dist, -1.0), 1.0);
                float inclination = acos(inclination_cos);

                if (inclination > PST_RAD_90 || (abs(inclination - PST_RAD_90) < 1e-30 && z_lrf <= 0)) {
                    float inclination_dist = (inclination - PST_RAD_135) / PST_RAD_90;
                    if (inclination > PST_RAD_135) {
                        init_weight += 1 - inclination_dist;
                    } else {
                        init_weight += 1 + inclination_dist;
                        ft[(desc_index + 1) * (n_bins_ + 1) + step_index] -= inclination_dist;
                    }
                } else {
                    float inclination_dist = (inclination - PST_RAD_45) / PST_RAD_90;
                    if (inclination < PST_RAD_45) {
                        init_weight += 1 + inclination_dist;
                    } else {
                        init_weight += 1 - inclination_dist;
                        ft[(desc_index - 1) * (n_bins_ + 1) + step_index] += inclination_dist;
                    }
                }

                if (y_lrf != 0.0 || x_lrf != 0.0) {
                    //Interpolation on the azimuth (adjacent horizontal volumes)
                    float azimuth = atan2(y_lrf, x_lrf);
                    int sel = desc_index >> 2;
                    float angular_sector_span = PST_RAD_45;
                    float angular_sector_start= - PST_RAD_PI_7_8;
                    float azimuth_dist = (azimuth - (angular_sector_start + angular_sector_span * sel)) / angular_sector_span;
                    azimuth_dist = max(-0.5, min(azimuth_dist, 0.5));
                    if (azimuth_dist > 0) {
                        init_weight += 1 - azimuth_dist;
                        int interp_index = (desc_index + 4) % max_angular_sectors_;
                        ft[interp_index * (n_bins_ + 1) + step_index] += azimuth_dist;
                    } else {
                        int interp_index = (desc_index - 4 + max_angular_sectors_) % max_angular_sectors_;
                        init_weight += 1 + azimuth_dist;
                        ft[interp_index * (n_bins_ + 1) + step_index] -= azimuth_dist;
                    }
                }
                ft[volume_index + step_index] += init_weight;
            }
            const float ftnorm = ft.norm();
            if (ftnorm > 0) {
                ft /= ftnorm;
            }
            return ft;
        }
    };
}

std::shared_ptr<Feature<352>> ComputeSHOTFeature(
        const geometry::PointCloud &input,
        float radius,
        const geometry::KDTreeSearchParam &search_param) {
    auto feature = std::make_shared<Feature<352>>();
    feature->Resize((int)input.points_.size());

    geometry::KDTreeFlann kdtree(input);
    utility::device_vector<int> indices;
    utility::device_vector<float> distance2;
    int knn;
    switch (search_param.GetSearchType()) {
        case geometry::KDTreeSearchParam::SearchType::Knn:
            knn = ((const geometry::KDTreeSearchParamKNN &)search_param).knn_;
            break;
        case geometry::KDTreeSearchParam::SearchType::Radius:
            knn = ((const geometry::KDTreeSearchParamRadius &)search_param)
                          .max_nn_;
            break;
        default:
            utility::LogError("Unsupport search param type.");
            return feature;
    }
    kdtree.Search(input.points_, search_param, indices, distance2);
    compute_shot_functor func(thrust::raw_pointer_cast(input.points_.data()),
                              thrust::raw_pointer_cast(input.normals_.data()),
                              thrust::raw_pointer_cast(indices.data()),
                              thrust::raw_pointer_cast(distance2.data()),
                              radius, knn);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(input.points_.size()),
                      feature->data_.begin(), func);
    return feature;
}

}
}