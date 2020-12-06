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
#include "cupoch_pybind/io/io.h"

using namespace cupoch;

void pybind_ros_io(py::module &m_io) {
    py::class_<io::PointField> point_field(m_io, "PointField");
    point_field.def(py::init<const std::string&, int, uint8_t, int>())
               .def_readwrite("name", &io::PointField::name_)
               .def_readwrite("offset", &io::PointField::offset_)
               .def_readwrite("datatype", &io::PointField::datatype_)
               .def_readwrite("count", &io::PointField::count_);

    py::class_<io::PointCloud2MsgInfo> info(m_io, "PointCloud2MsgInfo");
    info.def(py::init<int, int, const std::vector<io::PointField>, bool, int, int, bool>())
               .def_readwrite("width", &io::PointCloud2MsgInfo::width_)
               .def_readwrite("height", &io::PointCloud2MsgInfo::height_)
               .def_readwrite("fields", &io::PointCloud2MsgInfo::fields_)
               .def_readwrite("is_bigendian", &io::PointCloud2MsgInfo::is_bigendian_)
               .def_readwrite("point_step", &io::PointCloud2MsgInfo::point_step_)
               .def_readwrite("row_step", &io::PointCloud2MsgInfo::row_step_)
               .def_readwrite("is_dense", &io::PointCloud2MsgInfo::is_dense_)
               .def_static("default", &io::PointCloud2MsgInfo::Default);

    m_io.def("create_from_pointcloud2_msg",
             [] (const py::bytes &bytes, const io::PointCloud2MsgInfo& info) {
                 py::buffer_info binfo(py::buffer(bytes).request());
                 const uint8_t *data = reinterpret_cast<const uint8_t *>(binfo.ptr);
                 size_t length = static_cast<size_t>(binfo.size);
                 return io::CreateFromPointCloud2Msg(data, length, info);
             });
    m_io.def("create_to_pointcloud2_msg",
             [] (const geometry::PointCloud& pointcloud) {
                 auto info = io::PointCloud2MsgInfo::Default(pointcloud.points_.size());
                 char *data = new char[info.row_step_];
                 io::CreateToPointCloud2Msg(reinterpret_cast<uint8_t *>(data), info, pointcloud);
                 return std::make_tuple(py::bytes(data, info.row_step_), info);
             });
}