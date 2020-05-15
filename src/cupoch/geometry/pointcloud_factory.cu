#include <Eigen/Dense>
#include <limits>

#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "cupoch/geometry/image.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/geometry/rgbdimage.h"
#include "cupoch/utility/console.h"
#include "cupoch/utility/helper.h"

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct depth_to_pointcloud_functor {
    depth_to_pointcloud_functor(
            const uint8_t *depth,
            const int width,
            int num_of_channels,
            int bytes_per_channel,
            const int stride,
            const thrust::pair<float, float> &principal_point,
            const thrust::pair<float, float> &focal_length,
            const Eigen::Matrix4f &camera_pose)
        : depth_(depth),
          width_(width),
          num_of_channels_(num_of_channels),
          bytes_per_channel_(bytes_per_channel),
          stride_(stride),
          principal_point_(principal_point),
          focal_length_(focal_length),
          camera_pose_(camera_pose){};
    const uint8_t *depth_;
    const int width_;
    const int num_of_channels_;
    const int bytes_per_channel_;
    const int stride_;
    const thrust::pair<float, float> principal_point_;
    const thrust::pair<float, float> focal_length_;
    const Eigen::Matrix4f camera_pose_;
    __device__ Eigen::Vector3f operator()(size_t idx) {
        int row = idx / width_;
        int col = idx % width_;
        const float d = *(float *)(&depth_[idx * num_of_channels_ *
                                           bytes_per_channel_ * stride_]);
        if (d <= 0.0) {
            return Eigen::Vector3f(std::numeric_limits<float>::infinity(),
                                   std::numeric_limits<float>::infinity(),
                                   std::numeric_limits<float>::infinity());
        } else {
            float z = d;
            float x = (col - principal_point_.first) * z / focal_length_.first;
            float y =
                    (row - principal_point_.second) * z / focal_length_.second;
            Eigen::Vector4f point =
                    camera_pose_ * Eigen::Vector4f(x, y, z, 1.0);
            return point.block<3, 1>(0, 0);
        }
    }
};

std::shared_ptr<PointCloud> CreatePointCloudFromFloatDepthImage(
        const Image &depth,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic,
        int stride) {
    auto pointcloud = std::make_shared<PointCloud>();
    const Eigen::Matrix4f camera_pose = extrinsic.inverse();
    const auto focal_length = intrinsic.GetFocalLength();
    const auto principal_point = intrinsic.GetPrincipalPoint();
    const size_t depth_size = depth.width_ * depth.height_;
    pointcloud->points_.resize(depth_size);
    depth_to_pointcloud_functor func(
            thrust::raw_pointer_cast(depth.data_.data()), depth.width_,
            depth.num_of_channels_, depth.bytes_per_channel_, stride,
            principal_point, focal_length, camera_pose);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator(depth_size),
                      pointcloud->points_.begin(), func);
    pointcloud->RemoveNoneFinitePoints(true, true);
    return pointcloud;
}

template <typename TC, int NC>
struct convert_from_rgbdimage_functor {
    convert_from_rgbdimage_functor(
            const uint8_t *depth,
            const uint8_t *color,
            int width,
            const Eigen::Matrix4f &camera_pose,
            const thrust::pair<float, float> &principal_point,
            const thrust::pair<float, float> &focal_length,
            float scale,
            bool project_valid_depth_only)
        : depth_(depth),
          color_(color),
          width_(width),
          camera_pose_(camera_pose),
          principal_point_(principal_point),
          focal_length_(focal_length),
          scale_(scale),
          project_valid_depth_only_(project_valid_depth_only){};
    const uint8_t *depth_;
    const uint8_t *color_;
    const int width_;
    const Eigen::Matrix4f camera_pose_;
    const thrust::pair<float, float> principal_point_;
    const thrust::pair<float, float> focal_length_;
    const float scale_;
    const bool project_valid_depth_only_;
    __device__ thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> operator()(
            size_t idx) const {
        int i = idx / width_;
        int j = idx % width_;
        float *p = (float *)(depth_ + idx * sizeof(float));
        TC *pc = (TC *)(color_ + idx * NC * sizeof(TC));
        if (*p > 0) {
            float z = (float)(*p);
            float x = (j - principal_point_.first) * z / focal_length_.first;
            float y = (i - principal_point_.second) * z / focal_length_.second;
            Eigen::Vector4f point =
                    camera_pose_ * Eigen::Vector4f(x, y, z, 1.0);
            Eigen::Vector3f points = point.block<3, 1>(0, 0);
            Eigen::Vector3f colors =
                    Eigen::Vector3f(pc[0], pc[(NC - 1) / 2], pc[NC - 1]) /
                    scale_;
            return thrust::make_tuple(points, colors);
        } else if (!project_valid_depth_only_) {
            float z = std::numeric_limits<float>::quiet_NaN();
            float x = std::numeric_limits<float>::quiet_NaN();
            float y = std::numeric_limits<float>::quiet_NaN();
            return thrust::make_tuple(
                    Eigen::Vector3f(x, y, z),
                    Eigen::Vector3f(std::numeric_limits<TC>::quiet_NaN(),
                                    std::numeric_limits<TC>::quiet_NaN(),
                                    std::numeric_limits<TC>::quiet_NaN()));
        } else {
            float z = std::numeric_limits<float>::infinity();
            float x = std::numeric_limits<float>::infinity();
            float y = std::numeric_limits<float>::infinity();
            return thrust::make_tuple(
                    Eigen::Vector3f(x, y, z),
                    Eigen::Vector3f(std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity()));
        }
    }
};

template <typename TC, int NC>
std::shared_ptr<PointCloud> CreatePointCloudFromRGBDImageT(
        const RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic,
        bool project_valid_depth_only) {
    auto pointcloud = std::make_shared<PointCloud>();
    Eigen::Matrix4f camera_pose = extrinsic.inverse();
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    float scale = (sizeof(TC) == 1) ? 255.0 : 1.0;
    int num_valid_pixels = image.depth_.height_ * image.depth_.width_;
    pointcloud->points_.resize(num_valid_pixels);
    pointcloud->colors_.resize(num_valid_pixels);
    convert_from_rgbdimage_functor<TC, NC> func(
            thrust::raw_pointer_cast(image.depth_.data_.data()),
            thrust::raw_pointer_cast(image.color_.data_.data()),
            image.depth_.width_, camera_pose, principal_point, focal_length,
            scale, project_valid_depth_only);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(num_valid_pixels),
                      make_tuple_begin(pointcloud->points_, pointcloud->colors_),
                      func);
    pointcloud->RemoveNoneFinitePoints(project_valid_depth_only, true);
    return pointcloud;
}

}  // namespace

std::shared_ptr<PointCloud> PointCloud::CreateFromDepthImage(
        const Image &depth,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic /* = Eigen::Matrix4f::Identity()*/,
        float depth_scale /* = 1000.0*/,
        float depth_trunc /* = 1000.0*/,
        int stride /* = 1*/) {
    if (depth.num_of_channels_ == 1) {
        if (depth.bytes_per_channel_ == 2) {
            auto float_depth =
                    depth.ConvertDepthToFloatImage(depth_scale, depth_trunc);
            return CreatePointCloudFromFloatDepthImage(*float_depth, intrinsic,
                                                       extrinsic, stride);
        } else if (depth.bytes_per_channel_ == 4) {
            return CreatePointCloudFromFloatDepthImage(depth, intrinsic,
                                                       extrinsic, stride);
        }
    }
    utility::LogError(
            "[CreatePointCloudFromDepthImage] Unsupported image format.");
    return std::make_shared<PointCloud>();
}

std::shared_ptr<PointCloud> PointCloud::CreateFromRGBDImage(
        const RGBDImage &image,
        const camera::PinholeCameraIntrinsic &intrinsic,
        const Eigen::Matrix4f &extrinsic /* = Eigen::Matrix4f::Identity()*/,
        bool project_valid_depth_only) {
    if (image.depth_.num_of_channels_ == 1 &&
        image.depth_.bytes_per_channel_ == 4) {
        if (image.color_.bytes_per_channel_ == 1 &&
            image.color_.num_of_channels_ == 3) {
            return CreatePointCloudFromRGBDImageT<uint8_t, 3>(
                    image, intrinsic, extrinsic, project_valid_depth_only);
        } else if (image.color_.bytes_per_channel_ == 4 &&
                   image.color_.num_of_channels_ == 1) {
            return CreatePointCloudFromRGBDImageT<float, 1>(
                    image, intrinsic, extrinsic, project_valid_depth_only);
        }
    }
    utility::LogError(
            "[CreatePointCloudFromRGBDImage] Unsupported image format.");
    return std::make_shared<PointCloud>();
}
