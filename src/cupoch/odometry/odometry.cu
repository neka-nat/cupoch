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
#include <Eigen/Dense>

#include "cupoch/geometry/image.h"
#include "cupoch/geometry/rgbdimage.h"
#include "cupoch/odometry/odometry.h"
#include "cupoch/odometry/rgbdodometry_jacobian.h"
#include "cupoch/utility/platform.h"

namespace cupoch {
namespace odometry {

namespace {

struct initialize_correspondence_map_functor {
    initialize_correspondence_map_functor(uint8_t *correspondence_map,
                                          uint8_t *depth_buffer,
                                          int width)
        : correspondence_map_(correspondence_map),
          depth_buffer_(depth_buffer),
          width_(width){};
    uint8_t *correspondence_map_;
    uint8_t *depth_buffer_;
    int width_;
    __device__ void operator()(size_t idx) {
        *(int *)(correspondence_map_ + idx * 2 * sizeof(int)) = -1;
        *(int *)(correspondence_map_ + (idx * 2 + 1) * sizeof(int)) = -1;
        *(float *)(depth_buffer_ + idx * sizeof(float)) = -1.0f;
    }
};

std::tuple<std::shared_ptr<geometry::Image>, std::shared_ptr<geometry::Image>>
InitializeCorrespondenceMap(int width, int height) {
    // initialization: filling with any (u,v) to (-1,-1)
    auto correspondence_map = std::make_shared<geometry::Image>();
    auto depth_buffer = std::make_shared<geometry::Image>();
    correspondence_map->Prepare(width, height, 2, 4);
    depth_buffer->Prepare(width, height, 1, 4);
    initialize_correspondence_map_functor func(
            thrust::raw_pointer_cast(correspondence_map->data_.data()),
            thrust::raw_pointer_cast(depth_buffer->data_.data()), width);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(width * height),
                     func);
    return std::make_tuple(correspondence_map, depth_buffer);
}

__device__ inline void AddElementToCorrespondenceMap(
        uint8_t *correspondence_map,
        uint8_t *depth_buffer,
        int width,
        int u_s,
        int v_s,
        int u_t,
        int v_t,
        float transformed_d_t) {
    int exist_u_t, exist_v_t;
    float exist_d_t;
    exist_u_t = *geometry::PointerAt<int>(correspondence_map, width, 2, u_s,
                                          v_s, 0);
    exist_v_t = *geometry::PointerAt<int>(correspondence_map, width, 2, u_s,
                                          v_s, 1);
    if (exist_u_t != -1 && exist_v_t != -1) {
        exist_d_t = *geometry::PointerAt<float>(depth_buffer, width, u_s, v_s);
        if (transformed_d_t <
            exist_d_t) {  // update nearer point as correspondence
            *geometry::PointerAt<int>(correspondence_map, width, 2, u_s, v_s,
                                      0) = u_t;
            *geometry::PointerAt<int>(correspondence_map, width, 2, u_s, v_s,
                                      1) = v_t;
            *geometry::PointerAt<float>(depth_buffer, width, u_s, v_s) =
                    transformed_d_t;
        }
    } else {  // register correspondence
        *geometry::PointerAt<int>(correspondence_map, width, 2, u_s, v_s, 0) =
                u_t;
        *geometry::PointerAt<int>(correspondence_map, width, 2, u_s, v_s, 1) =
                v_t;
        *geometry::PointerAt<float>(depth_buffer, width, u_s, v_s) =
                transformed_d_t;
    }
}

struct merge_correspondence_maps_functor {
    merge_correspondence_maps_functor(uint8_t *correspondence_map,
                                      uint8_t *depth_buffer,
                                      uint8_t *correspondence_map_part,
                                      uint8_t *depth_buffer_part,
                                      int width)
        : correspondence_map_(correspondence_map),
          depth_buffer_(depth_buffer),
          correspondence_map_part_(correspondence_map_part),
          depth_buffer_part_(depth_buffer_part),
          width_(width){};
    uint8_t *correspondence_map_;
    uint8_t *depth_buffer_;
    uint8_t *correspondence_map_part_;
    uint8_t *depth_buffer_part_;
    int width_;
    __device__ void operator()(size_t idx) {
        int v_s = idx / width_;
        int u_s = idx % width_;
        int u_t = *geometry::PointerAt<int>(correspondence_map_part_, width_, 2,
                                            u_s, v_s, 0);
        int v_t = *geometry::PointerAt<int>(correspondence_map_part_, width_, 2,
                                            u_s, v_s, 1);
        if (u_t != -1 && v_t != -1) {
            float transformed_d_t = *geometry::PointerAt<float>(
                    depth_buffer_part_, width_, u_s, v_s);
            AddElementToCorrespondenceMap(correspondence_map_, depth_buffer_,
                                          width_, u_s, v_s, u_t, v_t,
                                          transformed_d_t);
        }
    }
};

void MergeCorrespondenceMaps(geometry::Image &correspondence_map,
                             geometry::Image &depth_buffer,
                             geometry::Image &correspondence_map_part,
                             geometry::Image &depth_buffer_part) {
    merge_correspondence_maps_functor func(
            thrust::raw_pointer_cast(correspondence_map.data_.data()),
            thrust::raw_pointer_cast(depth_buffer.data_.data()),
            thrust::raw_pointer_cast(correspondence_map_part.data_.data()),
            thrust::raw_pointer_cast(depth_buffer_part.data_.data()),
            correspondence_map.width_);
    thrust::for_each(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(correspondence_map.width_ *
                                                   correspondence_map.height_),
            func);
}

struct compute_correspondence_map {
    compute_correspondence_map(const uint8_t *depth_s,
                               const uint8_t *depth_t,
                               int width,
                               int height,
                               uint8_t *correspondence_map,
                               uint8_t *depth_buffer,
                               const Eigen::Vector3f &Kt,
                               const Eigen::Matrix3f &KRK_inv,
                               float max_depth_diff)
        : depth_s_(depth_s),
          depth_t_(depth_t),
          width_(width),
          height_(height),
          correspondence_map_(correspondence_map),
          depth_buffer_(depth_buffer),
          Kt_(Kt),
          KRK_inv_(KRK_inv),
          max_depth_diff_(max_depth_diff){};
    const uint8_t *depth_s_;
    const uint8_t *depth_t_;
    int width_;
    int height_;
    uint8_t *correspondence_map_;
    uint8_t *depth_buffer_;
    const Eigen::Vector3f Kt_;
    const Eigen::Matrix3f KRK_inv_;
    const float max_depth_diff_;
    __device__ void operator()(size_t idx) {
        int v_s = idx / width_;
        int u_s = idx % width_;
        float d_s = *geometry::PointerAt<float>(depth_s_, width_, u_s, v_s);
        if (!isnan(d_s)) {
            Eigen::Vector3f uv_in_s =
                    d_s * KRK_inv_ * Eigen::Vector3f(u_s, v_s, 1.0) + Kt_;
            float transformed_d_s = uv_in_s(2);
            int u_t = (int)(uv_in_s(0) / transformed_d_s + 0.5);
            int v_t = (int)(uv_in_s(1) / transformed_d_s + 0.5);
            if (u_t >= 0 && u_t < width_ && v_t >= 0 && v_t < height_) {
                float d_t =
                        *geometry::PointerAt<float>(depth_t_, width_, u_t, v_t);
                if (!isnan(d_t) &&
                    std::abs(transformed_d_s - d_t) <= max_depth_diff_) {
                    AddElementToCorrespondenceMap(correspondence_map_,
                                                  depth_buffer_, width_, u_s,
                                                  v_s, u_t, v_t, (float)d_s);
                }
            }
        }
    }
};

struct compute_correspondence_functor {
    compute_correspondence_functor(const uint8_t *correspondence_map, int width)
        : correspondence_map_(correspondence_map), width_(width){};
    const uint8_t *correspondence_map_;
    const int width_;
    __device__ Eigen::Vector4i operator()(size_t idx) const {
        int v_s = idx / width_;
        int u_s = idx % width_;
        int u_t = *(int *)(correspondence_map_ + idx * 2 * sizeof(int));
        int v_t = *(int *)(correspondence_map_ + (idx * 2 + 1) * sizeof(int));
        return Eigen::Vector4i(u_s, v_s, u_t, v_t);
    }
};

void ComputeCorrespondence(const Eigen::Matrix3f intrinsic_matrix,
                           const Eigen::Matrix4f &extrinsic,
                           const geometry::Image &depth_s,
                           const geometry::Image &depth_t,
                           const OdometryOption &option,
                           CorrespondenceSetPixelWise &correspondence) {
    const Eigen::Matrix3f K = intrinsic_matrix;
    const Eigen::Matrix3f K_inv = K.inverse();
    const Eigen::Matrix3f R = extrinsic.block<3, 3>(0, 0);
    const Eigen::Matrix3f KRK_inv = K * R * K_inv;
    Eigen::Vector3f Kt = K * extrinsic.block<3, 1>(0, 3);

    std::shared_ptr<geometry::Image> correspondence_map;
    std::shared_ptr<geometry::Image> depth_buffer;
    std::tie(correspondence_map, depth_buffer) =
            InitializeCorrespondenceMap(depth_t.width_, depth_t.height_);

    std::shared_ptr<geometry::Image> correspondence_map_private;
    std::shared_ptr<geometry::Image> depth_buffer_private;
    std::tie(correspondence_map_private, depth_buffer_private) =
            InitializeCorrespondenceMap(depth_t.width_, depth_t.height_);

    compute_correspondence_map func_cm(
            thrust::raw_pointer_cast(depth_s.data_.data()),
            thrust::raw_pointer_cast(depth_t.data_.data()), depth_s.width_,
            depth_s.height_,
            thrust::raw_pointer_cast(correspondence_map_private->data_.data()),
            thrust::raw_pointer_cast(depth_buffer_private->data_.data()), Kt,
            KRK_inv, option.max_depth_diff_);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(depth_s.width_ *
                                                            depth_s.height_),
                     func_cm);

    MergeCorrespondenceMaps(*correspondence_map, *depth_buffer,
                            *correspondence_map_private, *depth_buffer_private);

    correspondence.resize(correspondence_map->width_ *
                          correspondence_map->height_);
    compute_correspondence_functor func_cc(
            thrust::raw_pointer_cast(correspondence_map->data_.data()),
            correspondence_map->width_);
    thrust::transform(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(correspondence_map->width_ *
                                                   correspondence_map->height_),
            correspondence.begin(), func_cc);
    auto end = thrust::remove_if(correspondence.begin(), correspondence.end(),
                                 [] __device__(const Eigen::Vector4i &pc) {
                                     return (pc[2] == -1 || pc[3] == -1);
                                 });
    correspondence.resize(thrust::distance(correspondence.begin(), end));
}

struct convert_depth_to_xyz_image_functor {
    convert_depth_to_xyz_image_functor(const uint8_t *depth,
                                       int width,
                                       uint8_t *image_xyz,
                                       float ox,
                                       float oy,
                                       float inv_fx,
                                       float inv_fy)
        : depth_(depth),
          width_(width),
          image_xyz_(image_xyz),
          ox_(ox),
          oy_(oy),
          inv_fx_(inv_fx),
          inv_fy_(inv_fy){};
    const uint8_t *depth_;
    const int width_;
    uint8_t *image_xyz_;
    const float ox_;
    const float oy_;
    const float inv_fx_;
    const float inv_fy_;
    __device__ void operator()(size_t idx) {
        int y = idx / width_;
        int x = idx % width_;
        float *px = geometry::PointerAt<float>(image_xyz_, width_, 3, x, y, 0);
        float *py = geometry::PointerAt<float>(image_xyz_, width_, 3, x, y, 1);
        float *pz = geometry::PointerAt<float>(image_xyz_, width_, 3, x, y, 2);
        float z = *geometry::PointerAt<float>(depth_, width_, x, y);
        *px = (x - ox_) * z * inv_fx_;
        *py = (y - oy_) * z * inv_fy_;
        *pz = z;
    }
};

std::shared_ptr<geometry::Image> ConvertDepthImageToXYZImage(
        const geometry::Image &depth, const Eigen::Matrix3f &intrinsic_matrix) {
    auto image_xyz = std::make_shared<geometry::Image>();
    if (depth.num_of_channels_ != 1 || depth.bytes_per_channel_ != 4) {
        utility::LogError(
                "[ConvertDepthImageToXYZImage] Unsupported image format.");
    }
    const float inv_fx = 1.0 / intrinsic_matrix(0, 0);
    const float inv_fy = 1.0 / intrinsic_matrix(1, 1);
    const float ox = intrinsic_matrix(0, 2);
    const float oy = intrinsic_matrix(1, 2);
    image_xyz->Prepare(depth.width_, depth.height_, 3, 4);

    convert_depth_to_xyz_image_functor func(
            thrust::raw_pointer_cast(depth.data_.data()), depth.width_,
            thrust::raw_pointer_cast(image_xyz->data_.data()), ox, oy, inv_fx,
            inv_fy);
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(image_xyz->width_ *
                                                            image_xyz->height_),
                     func);
    return image_xyz;
}

std::vector<Eigen::Matrix3f> CreateCameraMatrixPyramid(
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
        int levels) {
    std::vector<Eigen::Matrix3f> pyramid_camera_matrix;
    pyramid_camera_matrix.reserve(levels);
    for (int i = 0; i < levels; i++) {
        Eigen::Matrix3f level_camera_matrix;
        if (i == 0)
            level_camera_matrix = pinhole_camera_intrinsic.intrinsic_matrix_;
        else
            level_camera_matrix = 0.5 * pyramid_camera_matrix[i - 1];
        level_camera_matrix(2, 2) = 1.;
        pyramid_camera_matrix.push_back(level_camera_matrix);
    }
    return pyramid_camera_matrix;
}

struct compute_gtg_functor {
    compute_gtg_functor(const uint8_t *xyz_t, int width)
        : xyz_t_(xyz_t), width_(width){};
    const uint8_t *xyz_t_;
    const int width_;
    __device__ Eigen::Matrix6f operator()(const Eigen::Vector4i &corres) const {
        int u_t = corres(2);
        int v_t = corres(3);
        float x = *geometry::PointerAt<float>(xyz_t_, width_, 3, u_t, v_t, 0);
        float y = *geometry::PointerAt<float>(xyz_t_, width_, 3, u_t, v_t, 1);
        float z = *geometry::PointerAt<float>(xyz_t_, width_, 3, u_t, v_t, 2);
        Eigen::Vector6f g_r_1 =
                (Eigen::Vector6f() << 0.0, z, -y, 1.0, 0.0, 0.0).finished();
        Eigen::Vector6f g_r_2 =
                (Eigen::Vector6f() << -z, 0.0, x, 0.0, 1.0, 0.0).finished();
        Eigen::Vector6f g_r_3 =
                (Eigen::Vector6f() << y, -x, 0.0, 0.0, 0.0, 1.0).finished();
        return g_r_1 * g_r_1.transpose() + g_r_2 * g_r_2.transpose() +
               g_r_3 * g_r_3.transpose();
    }
};

Eigen::Matrix6f CreateInformationMatrix(
        const Eigen::Matrix4f &extrinsic,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
        const geometry::Image &depth_s,
        const geometry::Image &depth_t,
        const OdometryOption &option) {
    CorrespondenceSetPixelWise correspondence;
    ComputeCorrespondence(pinhole_camera_intrinsic.intrinsic_matrix_, extrinsic,
                          depth_s, depth_t, option, correspondence);

    auto xyz_t = ConvertDepthImageToXYZImage(
            depth_t, pinhole_camera_intrinsic.intrinsic_matrix_);

    // write q^*
    // see http://redwood-data.org/indoor/registration.html
    // note: I comes first and q_skew is scaled by factor 2.
    compute_gtg_functor func(thrust::raw_pointer_cast(xyz_t->data_.data()),
                             xyz_t->width_);
    Eigen::Matrix6f init = Eigen::Matrix6f::Identity();
    Eigen::Matrix6f GTG = thrust::transform_reduce(
            utility::exec_policy(0)->on(0),
            correspondence.begin(), correspondence.end(), func, init,
            thrust::plus<Eigen::Matrix6f>());
    return GTG;
}

struct make_correspondence_pixel_pair {
    make_correspondence_pixel_pair(const uint8_t *image_s,
                                   const uint8_t *image_t,
                                   int width)
        : image_s_(image_s), image_t_(image_t), width_(width){};
    const uint8_t *image_s_;
    const uint8_t *image_t_;
    int width_;
    __device__ thrust::tuple<float, float> operator()(
            const Eigen::Vector4i &corres) const {
        int u_s = corres(0);
        int v_s = corres(1);
        int u_t = corres(2);
        int v_t = corres(3);
        return thrust::make_tuple(
                *geometry::PointerAt<float>(image_s_, width_, u_s, v_s),
                *geometry::PointerAt<float>(image_t_, width_, u_t, v_t));
    }
};

void NormalizeIntensity(geometry::Image &image_s,
                        geometry::Image &image_t,
                        CorrespondenceSetPixelWise &correspondence) {
    if (image_s.width_ != image_t.width_ ||
        image_s.height_ != image_t.height_) {
        utility::LogError(
                "[NormalizeIntensity] Size of two input images should be "
                "same");
    }
    make_correspondence_pixel_pair func_tf(
            thrust::raw_pointer_cast(image_s.data_.data()),
            thrust::raw_pointer_cast(image_t.data_.data()), image_s.width_);
    auto means = thrust::transform_reduce(
            utility::exec_policy(0)->on(0),
            correspondence.begin(), correspondence.end(), func_tf,
            thrust::make_tuple(0.0f, 0.0f), add_tuple_functor<float, float>());
    float mean_s = thrust::get<0>(means) / (float)correspondence.size();
    float mean_t = thrust::get<1>(means) / (float)correspondence.size();
    image_s.LinearTransform(0.5 / mean_s, 0.0);
    image_t.LinearTransform(0.5 / mean_t, 0.0);
}

inline std::shared_ptr<geometry::RGBDImage> PackRGBDImage(
        const geometry::Image &color, const geometry::Image &depth) {
    return std::make_shared<geometry::RGBDImage>(
            geometry::RGBDImage(color, depth));
}

struct preprocess_depth_functor {
    preprocess_depth_functor(uint8_t *depth, float min_depth, float max_depth)
        : depth_(depth), min_depth_(min_depth), max_depth_(max_depth){};
    uint8_t *depth_;
    const float min_depth_;
    const float max_depth_;
    __device__ void operator()(size_t idx) {
        float *p = (float *)(depth_ + idx * sizeof(float));
        if ((*p < min_depth_ || *p > max_depth_ || *p <= 0))
            *p = std::numeric_limits<float>::quiet_NaN();
    }
};

std::shared_ptr<geometry::Image> PreprocessDepth(
        cudaStream_t stream,
        const geometry::Image &depth_orig,
        const OdometryOption &option) {
    std::shared_ptr<geometry::Image> depth_processed =
            std::make_shared<geometry::Image>();
    *depth_processed = depth_orig;
    preprocess_depth_functor func(
            thrust::raw_pointer_cast(depth_processed->data_.data()),
            option.min_depth_, option.max_depth_);
    thrust::for_each(
            utility::exec_policy(stream)->on(stream),
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(depth_processed->width_ *
                                                   depth_processed->height_),
            func);
    return depth_processed;
}

inline bool CheckImagePair(const geometry::Image &image_s,
                           const geometry::Image &image_t) {
    return (image_s.width_ == image_t.width_ &&
            image_s.height_ == image_t.height_);
}

inline bool CheckRGBDImagePair(const geometry::RGBDImage &source,
                               const geometry::RGBDImage &target) {
    return (CheckImagePair(source.color_, target.color_) &&
            CheckImagePair(source.depth_, target.depth_) &&
            CheckImagePair(source.color_, source.depth_) &&
            CheckImagePair(target.color_, target.depth_) &&
            source.color_.num_of_channels_ == 1 &&
            source.depth_.num_of_channels_ == 1 &&
            target.color_.num_of_channels_ == 1 &&
            target.depth_.num_of_channels_ == 1 &&
            source.color_.bytes_per_channel_ == 4 &&
            target.color_.bytes_per_channel_ == 4 &&
            source.depth_.bytes_per_channel_ == 4 &&
            target.depth_.bytes_per_channel_ == 4);
}

std::tuple<std::shared_ptr<geometry::RGBDImage>,
           std::shared_ptr<geometry::RGBDImage>>
InitializeRGBDOdometry(
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
        const Eigen::Matrix4f &odo_init,
        const OdometryOption &option) {
    auto source_gray =
            source.color_.Filter(geometry::Image::FilterType::Gaussian3);
    auto target_gray =
            target.color_.Filter(geometry::Image::FilterType::Gaussian3);
    auto source_depth_preprocessed =
            PreprocessDepth(utility::GetStream(0), source.depth_, option);
    auto target_depth_preprocessed =
            PreprocessDepth(utility::GetStream(1), target.depth_, option);
    cudaSafeCall(cudaDeviceSynchronize());
    auto source_depth = source_depth_preprocessed->Filter(
            geometry::Image::FilterType::Gaussian3);
    auto target_depth = target_depth_preprocessed->Filter(
            geometry::Image::FilterType::Gaussian3);

    CorrespondenceSetPixelWise correspondence;
    ComputeCorrespondence(pinhole_camera_intrinsic.intrinsic_matrix_, odo_init,
                          *source_depth, *target_depth, option, correspondence);
    NormalizeIntensity(*source_gray, *target_gray, correspondence);

    auto source_out = PackRGBDImage(*source_gray, *source_depth);
    auto target_out = PackRGBDImage(*target_gray, *target_depth);
    return std::make_tuple(source_out, target_out);
}

template <typename JacobianType>
struct compute_jacobian_and_residual_functor
    : public utility::multiple_jacobians_residuals_functor<Eigen::Vector6f, 2> {
    compute_jacobian_and_residual_functor(const uint8_t *source_color,
                                          const uint8_t *source_depth,
                                          const uint8_t *target_color,
                                          const uint8_t *target_depth,
                                          const uint8_t *source_xyz,
                                          const uint8_t *target_dx_color,
                                          const uint8_t *target_dx_depth,
                                          const uint8_t *target_dy_color,
                                          const uint8_t *target_dy_depth,
                                          int width,
                                          const Eigen::Matrix3f &intrinsic,
                                          const Eigen::Matrix4f &extrinsic,
                                          const Eigen::Vector4i *corresps)
        : source_color_(source_color),
          source_depth_(source_depth),
          target_color_(target_color),
          target_depth_(target_depth),
          source_xyz_(source_xyz),
          target_dx_color_(target_dx_color),
          target_dx_depth_(target_dx_depth),
          target_dy_color_(target_dy_color),
          target_dy_depth_(target_dy_depth),
          width_(width),
          intrinsic_(intrinsic),
          extrinsic_(extrinsic),
          corresps_(corresps){};
    const uint8_t *source_color_;
    const uint8_t *source_depth_;
    const uint8_t *target_color_;
    const uint8_t *target_depth_;
    const uint8_t *source_xyz_;
    const uint8_t *target_dx_color_;
    const uint8_t *target_dx_depth_;
    const uint8_t *target_dy_color_;
    const uint8_t *target_dy_depth_;
    const int width_;
    const Eigen::Matrix3f intrinsic_;
    const Eigen::Matrix4f extrinsic_;
    const Eigen::Vector4i *corresps_;
    JacobianType jacobian_;
    __device__ void operator()(int i,
                               Eigen::Vector6f J_r[2],
                               float r[2]) const {
        jacobian_.ComputeJacobianAndResidual(
                i, J_r, r, source_color_, source_depth_, target_color_,
                target_depth_, source_xyz_, target_dx_color_, target_dx_depth_,
                target_dy_color_, target_dy_depth_, width_, intrinsic_,
                extrinsic_, corresps_);
    }
};

template <typename JacobianType>
std::tuple<bool, Eigen::Matrix4f> DoSingleIteration(
        int iter,
        int level,
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const geometry::Image &source_xyz,
        const geometry::RGBDImage &target_dx,
        const geometry::RGBDImage &target_dy,
        const Eigen::Matrix3f &intrinsic,
        const Eigen::Matrix4f &extrinsic_initial,
        const OdometryOption &option) {
    CorrespondenceSetPixelWise correspondence;
    ComputeCorrespondence(intrinsic, extrinsic_initial, source.depth_,
                          target.depth_, option, correspondence);
    int corresps_count = (int)correspondence.size();

    compute_jacobian_and_residual_functor<JacobianType> func(
            thrust::raw_pointer_cast(source.color_.data_.data()),
            thrust::raw_pointer_cast(source.depth_.data_.data()),
            thrust::raw_pointer_cast(target.color_.data_.data()),
            thrust::raw_pointer_cast(target.depth_.data_.data()),
            thrust::raw_pointer_cast(source_xyz.data_.data()),
            thrust::raw_pointer_cast(target_dx.color_.data_.data()),
            thrust::raw_pointer_cast(target_dx.depth_.data_.data()),
            thrust::raw_pointer_cast(target_dy.color_.data_.data()),
            thrust::raw_pointer_cast(target_dy.depth_.data_.data()),
            source.color_.width_, intrinsic, extrinsic_initial,
            thrust::raw_pointer_cast(correspondence.data()));
    utility::LogDebug("Iter : {:d}, Level : {:d}, ", iter, level);
    Eigen::Matrix6f JTJ;
    Eigen::Vector6f JTr;
    float r2;
    thrust::tie(JTJ, JTr, r2) =
            utility::ComputeJTJandJTr<Eigen::Matrix6f, Eigen::Vector6f, 2>(
                    func, corresps_count);

    bool is_success;
    Eigen::Matrix4f extrinsic;
    thrust::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);
    if (!is_success) {
        utility::LogWarning("[ComputeOdometry] no solution!");
        return std::make_tuple(false, Eigen::Matrix4f::Identity());
    } else {
        return std::make_tuple(true, extrinsic);
    }
}

struct weight_reduce_functor {
    weight_reduce_functor(float sigma2, float nu) : sigma2_(sigma2), nu_(nu){};
    const float sigma2_;
    const float nu_;
    __device__ float operator()(float r2) const {
        return r2 * (nu_ + 1.0) / (nu_ + r2 / sigma2_);
    }
};

struct calc_weights_functor {
    calc_weights_functor(float nu) : nu_(nu){};
    const float nu_;
    __device__ float operator()(float r2, float w_sum) const {
        return (nu_ + 1) / (nu_ + r2 / w_sum);
    }
};

template <typename JacobianType>
std::tuple<bool, Eigen::Matrix4f, float> DoSingleIterationWeighted(
        int iter,
        int level,
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const geometry::Image &source_xyz,
        const geometry::RGBDImage &target_dx,
        const geometry::RGBDImage &target_dy,
        const Eigen::Matrix3f &intrinsic,
        const Eigen::Matrix4f &extrinsic_initial,
        const Eigen::Vector6f &prev_twist,
        const Eigen::Vector6f &curr_vel,
        const OdometryOption &option,
        float sigma2) {
    CorrespondenceSetPixelWise correspondence;
    ComputeCorrespondence(intrinsic, extrinsic_initial, source.depth_,
                          target.depth_, option, correspondence);
    int corresps_count = (int)correspondence.size();

    compute_jacobian_and_residual_functor<JacobianType> func(
            thrust::raw_pointer_cast(source.color_.data_.data()),
            thrust::raw_pointer_cast(source.depth_.data_.data()),
            thrust::raw_pointer_cast(target.color_.data_.data()),
            thrust::raw_pointer_cast(target.depth_.data_.data()),
            thrust::raw_pointer_cast(source_xyz.data_.data()),
            thrust::raw_pointer_cast(target_dx.color_.data_.data()),
            thrust::raw_pointer_cast(target_dx.depth_.data_.data()),
            thrust::raw_pointer_cast(target_dy.color_.data_.data()),
            thrust::raw_pointer_cast(target_dy.depth_.data_.data()),
            source.color_.width_, intrinsic, extrinsic_initial,
            thrust::raw_pointer_cast(correspondence.data()));
    utility::LogDebug("Iter : {:d}, Level : {:d}, ", iter, level);
    Eigen::Matrix6f JTJ;
    Eigen::Vector6f JTr;
    float r2;
    float sigma2_new;
    thrust::tie(JTJ, JTr, r2, sigma2_new) =
            utility::ComputeWeightedJTJandJTr<Eigen::Matrix6f, Eigen::Vector6f,
                                              2>(
                    func, weight_reduce_functor(sigma2, option.nu_),
                    calc_weights_functor(option.nu_), corresps_count);
    JTJ.diagonal() += option.inv_sigma_mat_diag_;
    JTr -= (option.inv_sigma_mat_diag_.array() *
            (prev_twist - curr_vel).array())
                   .matrix();
    bool is_success;
    Eigen::Matrix4f extrinsic;
    thrust::tie(is_success, extrinsic) =
            utility::SolveJacobianSystemAndObtainExtrinsicMatrix(JTJ, JTr);
    if (!is_success) {
        utility::LogWarning("[ComputeOdometry] no solution!");
        return std::make_tuple(false, Eigen::Matrix4f::Identity(), sigma2_new);
    } else {
        return std::make_tuple(true, extrinsic, sigma2_new);
    }
}

template <typename JacobianType>
std::tuple<bool, Eigen::Matrix4f> ComputeMultiscale(
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
        const Eigen::Matrix4f &extrinsic_initial,
        const OdometryOption &option) {
    std::vector<int> iter_counts = option.iteration_number_per_pyramid_level_;
    int num_levels = (int)iter_counts.size();

    auto source_pyramid = source.CreatePyramid(num_levels);
    auto target_pyramid = target.CreatePyramid(num_levels);
    auto target_pyramid_dx = geometry::RGBDImage::FilterPyramid(
            target_pyramid, geometry::Image::FilterType::Sobel3Dx);
    auto target_pyramid_dy = geometry::RGBDImage::FilterPyramid(
            target_pyramid, geometry::Image::FilterType::Sobel3Dy);

    Eigen::Matrix4f result_odo = extrinsic_initial.isZero()
                                         ? Eigen::Matrix4f::Identity()
                                         : extrinsic_initial;

    std::vector<Eigen::Matrix3f> pyramid_camera_matrix =
            CreateCameraMatrixPyramid(pinhole_camera_intrinsic,
                                      (int)iter_counts.size());

    for (int level = num_levels - 1; level >= 0; level--) {
        const Eigen::Matrix3f level_camera_matrix =
                pyramid_camera_matrix[level];

        auto source_xyz_level = ConvertDepthImageToXYZImage(
                source_pyramid[level]->depth_, level_camera_matrix);
        auto source_level = PackRGBDImage(source_pyramid[level]->color_,
                                          source_pyramid[level]->depth_);
        auto target_level = PackRGBDImage(target_pyramid[level]->color_,
                                          target_pyramid[level]->depth_);
        auto target_dx_level = PackRGBDImage(target_pyramid_dx[level]->color_,
                                             target_pyramid_dx[level]->depth_);
        auto target_dy_level = PackRGBDImage(target_pyramid_dy[level]->color_,
                                             target_pyramid_dy[level]->depth_);

        for (int iter = 0; iter < iter_counts[num_levels - level - 1]; iter++) {
            Eigen::Matrix4f curr_odo;
            bool is_success;
            std::tie(is_success, curr_odo) = DoSingleIteration<JacobianType>(
                    iter, level, *source_level, *target_level,
                    *source_xyz_level, *target_dx_level, *target_dy_level,
                    level_camera_matrix, result_odo, option);
            result_odo = curr_odo * result_odo;

            if (!is_success) {
                utility::LogWarning("[ComputeOdometry] no solution!");
                return std::make_tuple(false, Eigen::Matrix4f::Identity());
            }
        }
    }
    return std::make_tuple(true, result_odo);
}

template <typename JacobianType>
std::tuple<bool, Eigen::Matrix4f, Eigen::Vector6f> ComputeMultiscaleWeighted(
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
        const Eigen::Matrix4f &extrinsic_initial,
        const Eigen::Vector6f &prev_twist,
        const OdometryOption &option) {
    std::vector<int> iter_counts = option.iteration_number_per_pyramid_level_;
    int num_levels = (int)iter_counts.size();

    auto source_pyramid = source.CreatePyramid(num_levels);
    auto target_pyramid = target.CreatePyramid(num_levels);
    auto target_pyramid_dx = geometry::RGBDImage::FilterPyramid(
            target_pyramid, geometry::Image::FilterType::Sobel3Dx);
    auto target_pyramid_dy = geometry::RGBDImage::FilterPyramid(
            target_pyramid, geometry::Image::FilterType::Sobel3Dy);

    Eigen::Matrix4f result_odo = extrinsic_initial.isZero()
                                         ? Eigen::Matrix4f::Identity()
                                         : extrinsic_initial;

    std::vector<Eigen::Matrix3f> pyramid_camera_matrix =
            CreateCameraMatrixPyramid(pinhole_camera_intrinsic,
                                      (int)iter_counts.size());

    Eigen::Matrix4f curr_vel = Eigen::Matrix4f::Identity();
    float sigma2 = option.sigma2_init_;
    for (int level = num_levels - 1; level >= 0; level--) {
        const Eigen::Matrix3f level_camera_matrix =
                pyramid_camera_matrix[level];

        auto source_xyz_level = ConvertDepthImageToXYZImage(
                source_pyramid[level]->depth_, level_camera_matrix);
        auto source_level = PackRGBDImage(source_pyramid[level]->color_,
                                          source_pyramid[level]->depth_);
        auto target_level = PackRGBDImage(target_pyramid[level]->color_,
                                          target_pyramid[level]->depth_);
        auto target_dx_level = PackRGBDImage(target_pyramid_dx[level]->color_,
                                             target_pyramid_dx[level]->depth_);
        auto target_dy_level = PackRGBDImage(target_pyramid_dy[level]->color_,
                                             target_pyramid_dy[level]->depth_);
        for (int iter = 0; iter < iter_counts[num_levels - level - 1]; iter++) {
            Eigen::Matrix4f curr_odo;
            bool is_success;
            std::tie(is_success, curr_odo, sigma2) =
                    DoSingleIterationWeighted<JacobianType>(
                            iter, level, *source_level, *target_level,
                            *source_xyz_level, *target_dx_level,
                            *target_dy_level, level_camera_matrix, result_odo,
                            prev_twist,
                            utility::TransformMatrix4fToVector6f(curr_vel),
                            option, sigma2);
            curr_vel = curr_odo * curr_vel;
            result_odo = curr_odo * result_odo;

            if (!is_success) {
                utility::LogWarning("[ComputeOdometry] no solution!");
                return std::make_tuple(false, Eigen::Matrix4f::Identity(),
                                       Eigen::Vector6f::Zero());
            }
        }
    }
    return std::make_tuple(true, result_odo,
                           utility::TransformMatrix4fToVector6f(curr_vel));
}

template <typename JacobianType>
std::tuple<bool, Eigen::Matrix4f, Eigen::Vector6f, Eigen::Matrix6f>
ComputeRGBDOdometryT(
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
        const Eigen::Matrix4f &odo_init,
        const Eigen::Vector6f &prev_twist,
        const OdometryOption &option,
        bool is_weighted) {
    if (!CheckRGBDImagePair(source, target)) {
        utility::LogWarning(
                "[RGBDOdometry] Two RGBD pairs should be same in size.");
        return std::make_tuple(false, Eigen::Matrix4f::Identity(),
                               Eigen::Vector6f::Zero(),
                               Eigen::Matrix6f::Zero());
    }

    std::shared_ptr<geometry::RGBDImage> source_processed, target_processed;
    std::tie(source_processed, target_processed) = InitializeRGBDOdometry(
            source, target, pinhole_camera_intrinsic, odo_init, option);

    Eigen::Matrix4f extrinsic;
    Eigen::Vector6f twist = Eigen::Vector6f::Zero();
    bool is_success;
    if (is_weighted) {
        std::tie(is_success, extrinsic, twist) =
                ComputeMultiscaleWeighted<JacobianType>(
                        *source_processed, *target_processed,
                        pinhole_camera_intrinsic, odo_init, prev_twist, option);
    } else {
        std::tie(is_success, extrinsic) = ComputeMultiscale<JacobianType>(
                *source_processed, *target_processed, pinhole_camera_intrinsic,
                odo_init, option);
    }

    if (is_success) {
        Eigen::Matrix4f trans_output = extrinsic;
        Eigen::Matrix6f info_output = CreateInformationMatrix(
                extrinsic, pinhole_camera_intrinsic, source_processed->depth_,
                target_processed->depth_, option);
        return std::make_tuple(true, trans_output, twist, info_output);
    } else {
        return std::make_tuple(false, Eigen::Matrix4f::Identity(), twist,
                               Eigen::Matrix6f::Identity());
    }
}

template std::tuple<bool, Eigen::Matrix4f, Eigen::Vector6f, Eigen::Matrix6f>
ComputeRGBDOdometryT<RGBDOdometryJacobianFromColorTerm>(
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
        const Eigen::Matrix4f &odo_init,
        const Eigen::Vector6f &prev_twist,
        const OdometryOption &option,
        bool is_weighted);

template std::tuple<bool, Eigen::Matrix4f, Eigen::Vector6f, Eigen::Matrix6f>
ComputeRGBDOdometryT<RGBDOdometryJacobianFromHybridTerm>(
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic,
        const Eigen::Matrix4f &odo_init,
        const Eigen::Vector6f &prev_twist,
        const OdometryOption &option,
        bool is_weighted);

}  // unnamed namespace

std::tuple<bool, Eigen::Matrix4f, Eigen::Matrix6f> ComputeRGBDOdometry(
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic
        /*= camera::PinholeCameraIntrinsic()*/,
        const Eigen::Matrix4f &odo_init /*= Eigen::Matrix4f::Identity()*/,
        const RGBDOdometryJacobian &jacobian_method
        /*=RGBDOdometryJacobianFromHybridTerm*/,
        const OdometryOption &option /*= OdometryOption()*/) {
    if (jacobian_method.jacobian_type_ == RGBDOdometryJacobian::COLOR_TERM) {
        auto res = ComputeRGBDOdometryT<RGBDOdometryJacobianFromColorTerm>(
                source, target, pinhole_camera_intrinsic, odo_init,
                Eigen::Vector6f::Zero(), option, false);
        return std::make_tuple(std::get<0>(res), std::get<1>(res),
                               std::get<3>(res));
    } else {
        auto res = ComputeRGBDOdometryT<RGBDOdometryJacobianFromHybridTerm>(
                source, target, pinhole_camera_intrinsic, odo_init,
                Eigen::Vector6f::Zero(), option, false);
        return std::make_tuple(std::get<0>(res), std::get<1>(res),
                               std::get<3>(res));
    }
}

std::tuple<bool, Eigen::Matrix4f, Eigen::Vector6f, Eigen::Matrix6f>
ComputeWeightedRGBDOdometry(
        const geometry::RGBDImage &source,
        const geometry::RGBDImage &target,
        const camera::PinholeCameraIntrinsic &pinhole_camera_intrinsic
        /*= camera::PinholeCameraIntrinsic()*/,
        const Eigen::Matrix4f &odo_init /*= Eigen::Matrix4f::Identity()*/,
        const Eigen::Vector6f &prev_twist /*= Eigen::Vector6f::Zero()*/,
        const RGBDOdometryJacobian &jacobian_method
        /*=RGBDOdometryJacobianFromHybridTerm*/,
        const OdometryOption &option /*= OdometryOption()*/) {
    return ComputeRGBDOdometryT<RGBDOdometryJacobianFromHybridTerm>(
            source, target, pinhole_camera_intrinsic, odo_init, prev_twist,
            option, true);
}

}  // namespace odometry
}  // namespace cupoch