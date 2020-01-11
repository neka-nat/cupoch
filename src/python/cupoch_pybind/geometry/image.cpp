#include "cupoch/geometry/image.h"
#include "cupoch_pybind/docstring.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/geometry/geometry_trampoline.h"
#include "cupoch/utility/platform.h"

using namespace cupoch;

void pybind_image(py::module &m) {

    py::class_<geometry::Image, PyGeometry2D<geometry::Image>,
               std::shared_ptr<geometry::Image>, geometry::Geometry2D>
            image(m, "Image", py::buffer_protocol(),
                  "The image class stores image with customizable width, "
                  "height, num of channels and bytes per channel.");
    py::detail::bind_default_constructor<geometry::Image>(image);
    py::detail::bind_copy_functions<geometry::Image>(image);
    image.def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();
            int width, height, num_of_channels = 0, bytes_per_channel;
            if (info.format == py::format_descriptor<uint8_t>::format() ||
                info.format == py::format_descriptor<int8_t>::format()) {
                bytes_per_channel = 1;
            } else if (info.format ==
                               py::format_descriptor<uint16_t>::format() ||
                       info.format ==
                               py::format_descriptor<int16_t>::format()) {
                bytes_per_channel = 2;
            } else if (info.format == py::format_descriptor<float>::format()) {
                bytes_per_channel = 4;
            } else {
                throw std::runtime_error(
                        "Image can only be initialized from buffer of uint8, "
                        "uint16, or float!");
            }
            if (info.strides[info.ndim - 1] != bytes_per_channel) {
                throw std::runtime_error(
                        "Image can only be initialized from c-style buffer.");
            }
            if (info.ndim == 2) {
                num_of_channels = 1;
            } else if (info.ndim == 3) {
                num_of_channels = (int)info.shape[2];
            }
            height = (int)info.shape[0];
            width = (int)info.shape[1];
            auto img = new geometry::Image();
            img->Prepare(width, height, num_of_channels, bytes_per_channel);
            cudaSafeCall(cudaMemcpy(thrust::raw_pointer_cast(img->data_.data()), info.ptr,
                                    img->data_.size(), cudaMemcpyHostToDevice));
            return img;
         }))
            .def("__repr__",
                 [](const geometry::Image &img) {
                     return std::string("Image of size ") +
                            std::to_string(img.width_) + std::string("x") +
                            std::to_string(img.height_) + ", with " +
                            std::to_string(img.num_of_channels_) +
                            std::string(
                                    " channels.");
                 })
            .def("flip_vertical", &geometry::Image::FlipVertical,
                 "Function to flip image vertically (upside down)")
            .def("flip_horizontal", &geometry::Image::FlipHorizontal,
                 "Function to flip image horizontally (from left to right)");
}

void pybind_image_methods(py::module &m) {}