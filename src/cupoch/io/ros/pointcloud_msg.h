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
#pragma once
#include "cupoch/geometry/pointcloud.h"


namespace cupoch {
namespace io {

class PointField {
public:
    enum DataType {
        None = 0,
        Int8 = 1,
        UInt8 = 2,
        Int16 = 3,
        UInt16 = 4,
        Int32 = 5,
        UInt6 = 6,
        Float32 = 7,
        Float64 = 8,
    };
    PointField(const std::string& name,
               int offset,
               uint8_t datatype,
               int count)
    : name_(name), offset_(offset),
    datatype_(datatype), count_(count) {};
    ~PointField() {};

    std::string name_;
    int offset_;
    uint8_t datatype_;
    int count_;
};

class PointCloud2MsgInfo {
public:
    PointCloud2MsgInfo(int width, int height,
                       const std::vector<PointField>& fields,
                       bool is_bigendian,
                       int point_step,
                       int row_step,
                       bool is_dense = false)
    : width_(width), height_(height),
    fields_(fields), is_bigendian_(is_bigendian),
    point_step_(point_step), row_step_(row_step),
    is_dense_(is_dense) {};

    ~PointCloud2MsgInfo() {};

    static PointCloud2MsgInfo Default(int size, int point_step = 20) {
        return PointCloud2MsgInfo(size, 1,
                                  {PointField("x", 0, PointField::Float32, 1),
                                   PointField("y", 4, PointField::Float32, 1),
                                   PointField("z", 8, PointField::Float32, 1),
                                   PointField("rgb", 16, PointField::Float32, 1)},
                                  false,
                                  point_step,
                                  point_step * size,
                                  false);
    };

    static PointCloud2MsgInfo DefaultDense(int width, int height, int point_step = 32) {
        return PointCloud2MsgInfo(width, height,
                                  {PointField("x", 0, PointField::Float32, 1),
                                   PointField("y", 4, PointField::Float32, 1),
                                   PointField("z", 8, PointField::Float32, 1),
                                   PointField("rgb", 16, PointField::Float32, 1)},
                                  false,
                                  point_step,
                                  point_step * height,
                                  true);
    };

    int width_;
    int height_;
    std::vector<PointField> fields_;
    bool is_bigendian_;
    int point_step_;
    int row_step_;
    bool is_dense_;
};

std::shared_ptr<geometry::PointCloud> CreateFromPointCloud2Msg(
    const uint8_t* data, size_t size,
    const PointCloud2MsgInfo& info
);

void CreateToPointCloud2Msg(uint8_t* data, const PointCloud2MsgInfo& info, const geometry::PointCloud& pointcloud);

}
}