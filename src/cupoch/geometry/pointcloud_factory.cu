#include <Eigen/Dense>

#include "cupoch/camera/pinhole_camera_intrinsic.h"
#include "cupoch/geometry/image.h"
#include "cupoch/geometry/pointcloud.h"
#include "cupoch/utility/helper.h"
#include "cupoch/utility/console.h"
#include <limits>

using namespace cupoch;
using namespace cupoch::geometry;

namespace {

struct depth_to_pointcloud_functor {
    depth_to_pointcloud_functor(const uint8_t* depth, const int width,
                                int num_of_channels,
                                int bytes_per_channel, const int stride,
                                const thrust::pair<float, float>& principal_point,
                                const thrust::pair<float, float>& focal_length,
                                const Eigen::Matrix4f& camera_pose)
        : depth_(depth), width_(width), num_of_channels_(num_of_channels),
          bytes_per_channel_(bytes_per_channel), stride_(stride), principal_point_(principal_point),
          focal_length_(focal_length), camera_pose_(camera_pose) {};
    const uint8_t* depth_;
    const int width_;
    const int num_of_channels_;
    const int bytes_per_channel_;
    const int stride_;
    const thrust::pair<float, float> principal_point_;
    const thrust::pair<float, float> focal_length_;
    const Eigen::Matrix4f camera_pose_;
    __device__
    Eigen::Vector3f operator() (size_t idx) {
        int row = idx / width_;
        int col = idx % width_;
        const float d = *(float *)(&depth_[idx * num_of_channels_ * bytes_per_channel_ * stride_]);
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
    depth_to_pointcloud_functor func(thrust::raw_pointer_cast(depth.data_.data()),
                                     depth.width_, depth.num_of_channels_,
                                     depth.bytes_per_channel_, stride,
                                     principal_point, focal_length,
                                     camera_pose);
    thrust::transform(thrust::make_counting_iterator<size_t>(0), thrust::make_counting_iterator(depth_size),
                      pointcloud->points_.begin(), func);
    auto end = thrust::remove_if(pointcloud->points_.begin(), pointcloud->points_.end(),
                                 [] __device__ (const Eigen::Vector3f& pt) -> bool {return pt.array().isInf().any();});
    size_t n_result = thrust::distance(pointcloud->points_.begin(), end);
    pointcloud->points_.resize(n_result);
    return pointcloud;
}

}

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