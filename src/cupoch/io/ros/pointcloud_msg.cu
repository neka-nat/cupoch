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
#include "cupoch/io/ros/pointcloud_msg.h"
#include "cupoch/utility/platform.h"
#include "cupoch/utility/console.h"

namespace cupoch {
namespace io {

namespace {

struct convert_from_pointcloud2_msg_functor {
    convert_from_pointcloud2_msg_functor(const uint8_t* data,
                                         int point_x_offset,
                                         int point_y_offset,
                                         int point_z_offset,
                                         int rgb_offset,
                                         int point_step)
    : data_(data),
    point_x_offset_(point_x_offset),
    point_y_offset_(point_y_offset),
    point_z_offset_(point_z_offset),
    rgb_offset_(rgb_offset),
    point_step_(point_step) {};
    const uint8_t* data_;
    int point_x_offset_;
    int point_y_offset_;
    int point_z_offset_;
    int rgb_offset_;
    int point_step_;
    __device__
    thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> operator() (size_t idx) const {
        float x = *(float*)(data_ + idx * point_step_ + point_x_offset_);
        float y = *(float*)(data_ + idx * point_step_ + point_y_offset_);
        float z = *(float*)(data_ + idx * point_step_ + point_z_offset_);
        if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
            return thrust::make_tuple(Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity()),
                                      Eigen::Vector3f::Zero());
        }
        uint32_t rgb = *(uint32_t*)(data_ + idx * point_step_ + rgb_offset_);
        uint8_t r = (rgb & 0x00FF0000) >> 16;
        uint8_t g = (rgb & 0x0000FF00) >> 8;
        uint8_t b = (rgb & 0x000000FF);
        Eigen::Vector3f color = Eigen::Vector3f(r / 255.0, g / 255.0, b / 255.0);
        return thrust::make_tuple(Eigen::Vector3f(x, y, z), std::move(color));
    }
};

struct convert_to_pointcloud2_msg_functor {
    convert_to_pointcloud2_msg_functor(uint8_t* data, int point_step)
    : data_(data), point_step_(point_step) {}
    uint8_t *data_;
    int point_step_;
    __device__
    void operator() (const thrust::tuple<size_t, Eigen::Vector3f, Eigen::Vector3f>& x) {
        const size_t idx = thrust::get<0>(x);
        const Eigen::Vector4f point = (Eigen::Vector4f() << thrust::get<1>(x), 1.0).finished();
        const Eigen::Vector3f color = thrust::get<2>(x);
        memcpy(data_ + idx * point_step_, point.data(), 4 * sizeof(float));
        uint32_t r = (uint32_t)(color[0] * 255.0);
        uint32_t g = (uint32_t)(color[1] * 255.0);
        uint32_t b = (uint32_t)(color[2] * 255.0);
        uint32_t c = (r << 16) & (g << 8) & b;
        memcpy(data_ + idx * point_step_ + 16, &c, sizeof(uint32_t));
    }
};

int FindField(const std::vector<PointField>& fields, const std::string& ref) {
    for (int i = 0; i < fields.size(); ++i) {
        if (fields[i].name_ == ref) {
            return fields[i].offset_;
        }
    }
    return -1;
}

}

std::shared_ptr<geometry::PointCloud> CreateFromPointCloud2Msg(
    const uint8_t* data, size_t size,
    const PointCloud2MsgInfo& info
) {
    auto out = std::make_shared<geometry::PointCloud>();
    utility::device_vector<uint8_t> dv_data(size);
    cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(dv_data.data()), data, size, cudaMemcpyHostToDevice));
    int point_x_offset = FindField(info.fields_, "x");
    if (point_x_offset < 0) {
        utility::LogError("[CreateFromPointCloud2Msg] Field name 'x' is necessary.");
        return out;
    }
    int point_y_offset = FindField(info.fields_, "y");
    if (point_y_offset < 0) {
        utility::LogError("[CreateFromPointCloud2Msg] Field name 'y' is necessary.");
        return out;
    }
    int point_z_offset = FindField(info.fields_, "z");
    if (point_z_offset < 0) {
        utility::LogError("[CreateFromPointCloud2Msg] Field name 'z' is necessary.");
        return out;
    }
    int rgb_offset = FindField(info.fields_, "rgb");
    if (rgb_offset < 0) {
        utility::LogError("[CreateFromPointCloud2Msg] Field name 'rgb' is necessary.");
        return out;
    }
    convert_from_pointcloud2_msg_functor func(thrust::raw_pointer_cast(dv_data.data()),
                                              point_x_offset, point_y_offset, point_z_offset, rgb_offset,
                                              info.point_step_);
    resize_all(info.width_ * info.height_, out->points_, out->colors_);
    thrust::transform(thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(info.width_ * info.height_),
                      make_tuple_begin(out->points_, out->colors_), func);
    if (info.is_dense_) {
        out->RemoveNoneFinitePoints();
    }
    return out;
}

void CreateToPointCloud2Msg(uint8_t* data, const PointCloud2MsgInfo& info, const geometry::PointCloud& pointcloud) {
    if (!pointcloud.HasPoints()) {
        return;
    }
    if (info.width_ > 0 && info.point_step_ > 0 && info.row_step_ > 0) {
        utility::LogError("[CreateToPointCloud2Msg] Width and Step sizes must be greater than 0.");
        return;
    }
    if (info.height_ != 1) {
        utility::LogError("[CreateToPointCloud2Msg] Height must be 1.");
        return;
    }
    utility::device_vector<uint8_t> dv_data(info.row_step_);
    convert_to_pointcloud2_msg_functor func(thrust::raw_pointer_cast(dv_data.data()), info.point_step_);
    thrust::for_each(enumerate_begin(pointcloud.points_, pointcloud.colors_),
                     enumerate_end(pointcloud.points_, pointcloud.colors_), func);
    cudaSafeCall(cudaMemcpy(data, thrust::raw_pointer_cast(dv_data.data()), info.row_step_, cudaMemcpyDeviceToHost));
}

}
}