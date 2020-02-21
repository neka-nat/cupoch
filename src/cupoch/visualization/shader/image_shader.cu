#include "cupoch/visualization/shader/image_shader.h"

#include <algorithm>

#include "cupoch/geometry/image.h"
#include "cupoch/visualization/shader/shader.h"
#include "cupoch/visualization/utility/color_map.h"
#include "cupoch/utility/range.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace cupoch;
using namespace cupoch::visualization;
using namespace cupoch::visualization::glsl;

namespace {

__device__
uint8_t ConvertColorFromFloatToUnsignedChar(float color) {
    if (std::isnan(color)) {
        return 0;
    } else {
        thrust::minimum<float> min;
        thrust::maximum<float> max;
        float unified_color = min(1.0f, max(0.0f, color));
        return (uint8_t)(unified_color * 255.0f);
    }
}

struct copy_float_gray_image_functor {
    copy_float_gray_image_functor(const uint8_t* gray) : gray_(gray) {};
    const uint8_t* gray_;
    __device__
    uint8_t operator() (size_t k) const {
        int idx = k / 3;
        float *p = (float *)(gray_ + idx * 4);
        uint8_t color = ConvertColorFromFloatToUnsignedChar(*p);
        return color;
    }
};

struct copy_float_rgb_image_functor {
    copy_float_rgb_image_functor(const uint8_t* rgb) : rgb_(rgb) {};
    const uint8_t* rgb_;
    __device__
    uint8_t operator() (size_t idx) const {
        float *p = (float *)(rgb_ + idx * 4);
        return ConvertColorFromFloatToUnsignedChar(*p);
    }
};

struct copy_int16_rgb_image_functor {
    copy_int16_rgb_image_functor(const uint8_t* rgb) : rgb_(rgb) {};
    const uint8_t* rgb_;
    __device__
    uint8_t operator() (size_t idx) const {
        uint16_t *p = (uint16_t *)(rgb_ + idx * 2);
        return (uint8_t)((*p) & 0xff);
    }
};

struct copy_depth_image_functor {
    copy_depth_image_functor(const uint8_t* depth, int max_depth)
        : depth_(depth), max_depth_(max_depth) {};
    const uint8_t* depth_;
    const int max_depth_;
    const ColorMap::ColorMapOption colormap_option_ = GetGlobalColorMapOption();
    __device__
    uint8_t operator() (size_t k) const {
        thrust::minimum<float> min;
        int i = k / 3;
        int j = k % 3;
        uint16_t *p = (uint16_t *)(depth_ + i * 2);
        float depth = min(float(*p) / float(max_depth_), 1.0);
        Eigen::Vector3f color = GetColorMapColor(depth, colormap_option_);
        return (uint8_t)(color(j) * 255);
    }
};

}  // unnamed namespace

bool ImageShader::Compile() {
    if (CompileShaders(image_vertex_shader, NULL, image_fragment_shader) == false) {
        PrintShaderWarning("Compiling shaders failed.");
        return false;
    }
    vertex_position_ = glGetAttribLocation(program_, "vertex_position");
    vertex_UV_ = glGetAttribLocation(program_, "vertex_UV");
    image_texture_ = glGetUniformLocation(program_, "image_texture");
    vertex_scale_ = glGetUniformLocation(program_, "vertex_scale");
    return true;
}

void ImageShader::Release() {
    UnbindGeometry();
    ReleaseProgram();
}

bool ImageShader::BindGeometry(const geometry::Geometry &geometry,
                               const RenderOption &option,
                               const ViewControl &view) {
    // If there is already geometry, we first unbind it.
    // We use GL_STATIC_DRAW. When geometry changes, we clear buffers and
    // rebind the geometry. Note that this approach is slow. If the geometry is
    // changing per frame, consider implementing a new ShaderWrapper using
    // GL_STREAM_DRAW, and replace UnbindGeometry() with Buffer Object
    // Streaming mechanisms.
    UnbindGeometry();

    // Prepare data to be passed to GPU
    const size_t num_data_height = GetDataHeight(geometry);
    const size_t num_data_width = GetDataWidth(geometry);

    // Create buffers and bind the geometry
    const GLfloat vertex_position_buffer_data[18] = {
            -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f, 1.0f, 1.0f,  0.0f, -1.0f, 1.0f, 0.0f,
    };
    const GLfloat vertex_UV_buffer_data[12] = {
            0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    };
    glGenBuffers(1, &vertex_position_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_position_buffer_data),
                 vertex_position_buffer_data, GL_STATIC_DRAW);
    glGenBuffers(1, &vertex_UV_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_UV_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_UV_buffer_data),
                 vertex_UV_buffer_data, GL_STATIC_DRAW);

    glGenTextures(1, &image_texture_buffer_);
    glBindTexture(GL_TEXTURE_2D, image_texture_buffer_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, num_data_width,
                 num_data_height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

    if (option.interpolation_option_ ==
        RenderOption::TextureInterpolationOption::Nearest) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    } else {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    glGenBuffers(1, &image_pixel_buffer_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image_pixel_buffer_);
    size_t data_size = GetDataSize(geometry);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, data_size, 0, GL_STATIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaSafeCall(cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources_[0], image_pixel_buffer_, cudaGraphicsMapFlagsNone));
    uint8_t* raw_render_image_ptr;
    size_t n_bytes;
    cudaSafeCall(cudaGraphicsMapResources(1, cuda_graphics_resources_));
    cudaSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&raw_render_image_ptr, &n_bytes, cuda_graphics_resources_[0]));
    thrust::device_ptr<uint8_t> dev_render_image_ptr = thrust::device_pointer_cast(raw_render_image_ptr);
    if (PrepareBinding(geometry, option, view, dev_render_image_ptr) == false) {
        PrintShaderWarning("Binding failed when preparing data.");
        return false;
    }

    Unmap(1);
    bound_ = true;
    return true;
}

bool ImageShader::RenderGeometry(const geometry::Geometry &geometry,
                                 const RenderOption &option,
                                 const ViewControl &view) {
    if (PrepareRendering(geometry, option, view) == false) {
        PrintShaderWarning("Rendering failed during preparation.");
        return false;
    }

    const size_t num_data_height = GetDataHeight(geometry);
    const size_t num_data_width = GetDataWidth(geometry);
    glUseProgram(program_);
    glUniform3fv(vertex_scale_, 1, vertex_scale_data_.data());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, image_texture_buffer_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image_pixel_buffer_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, num_data_width, num_data_height,
                    GL_RGB, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glUniform1i(image_texture_, 0);
    glEnableVertexAttribArray(vertex_position_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_position_buffer_);
    glVertexAttribPointer(vertex_position_, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    glEnableVertexAttribArray(vertex_UV_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_UV_buffer_);
    glVertexAttribPointer(vertex_UV_, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    glDrawArrays(draw_arrays_mode_, 0, draw_arrays_size_);
    glDisableVertexAttribArray(vertex_position_);
    glDisableVertexAttribArray(vertex_UV_);

    return true;
}

void ImageShader::UnbindGeometry() {
    if (bound_) {
        cudaSafeCall(cudaGraphicsUnregisterResource(cuda_graphics_resources_[0]));
        glDeleteBuffers(1, &image_pixel_buffer_);
        glDeleteBuffers(1, &vertex_position_buffer_);
        glDeleteBuffers(1, &vertex_UV_buffer_);
        glDeleteTextures(1, &image_texture_buffer_);
        bound_ = false;
    }
}

bool ImageShaderForImage::PrepareRendering(const geometry::Geometry &geometry,
                                           const RenderOption &option,
                                           const ViewControl &view) {
    if (geometry.GetGeometryType() != geometry::Geometry::GeometryType::Image) {
        PrintShaderWarning("Rendering type is not geometry::Image.");
        return false;
    }
    const geometry::Image &image = (const geometry::Image &)geometry;
    GLfloat ratio_x, ratio_y;
    switch (option.image_stretch_option_) {
        case RenderOption::ImageStretchOption::StretchKeepRatio:
            ratio_x = GLfloat(image.width_) / GLfloat(view.GetWindowWidth());
            ratio_y = GLfloat(image.height_) / GLfloat(view.GetWindowHeight());
            if (ratio_x < ratio_y) {
                ratio_x /= ratio_y;
                ratio_y = 1.0f;
            } else {
                ratio_y /= ratio_x;
                ratio_x = 1.0f;
            }
            break;
        case RenderOption::ImageStretchOption::StretchWithWindow:
            ratio_x = 1.0f;
            ratio_y = 1.0f;
            break;
        case RenderOption::ImageStretchOption::OriginalSize:
        default:
            ratio_x = GLfloat(image.width_) / GLfloat(view.GetWindowWidth());
            ratio_y = GLfloat(image.height_) / GLfloat(view.GetWindowHeight());
            break;
    }
    vertex_scale_data_(0) = ratio_x;
    vertex_scale_data_(1) = ratio_y;
    vertex_scale_data_(2) = 1.0f;
    glDisable(GL_DEPTH_TEST);
    return true;
}

bool ImageShaderForImage::PrepareBinding(const geometry::Geometry &geometry,
                                         const RenderOption &option,
                                         const ViewControl &view,
                                         thrust::device_ptr<uint8_t> &render_image) {
    if (geometry.GetGeometryType() != geometry::Geometry::GeometryType::Image) {
        PrintShaderWarning("Rendering type is not geometry::Image.");
        return false;
    }
    const geometry::Image &image = (const geometry::Image &)geometry;
    if (image.HasData() == false) {
        PrintShaderWarning("Binding failed with empty image.");
        return false;
    }

    if (image.num_of_channels_ == 3 && image.bytes_per_channel_ == 1) {
        thrust::copy(image.data_.begin(), image.data_.end(), render_image);
    } else {
        if (image.num_of_channels_ == 1 && image.bytes_per_channel_ == 1) {
            // grayscale image
            thrust::repeated_range<utility::device_vector<uint8_t>::const_iterator> range(image.data_.begin(), image.data_.end(), 3);
            thrust::copy(range.begin(), range.end(), render_image);
        } else if (image.num_of_channels_ == 1 &&
                   image.bytes_per_channel_ == 4) {
            // grayscale image with floating point per channel
            copy_float_gray_image_functor func(thrust::raw_pointer_cast(image.data_.data()));
            thrust::transform(thrust::make_counting_iterator<size_t>(0),
                              thrust::make_counting_iterator<size_t>(image.height_ * image.width_ * 3),
                              render_image, func);
        } else if (image.num_of_channels_ == 3 &&
                   image.bytes_per_channel_ == 4) {
            // RGB image with floating point per channel
            copy_float_rgb_image_functor func(thrust::raw_pointer_cast(image.data_.data()));
            thrust::transform(thrust::make_counting_iterator<size_t>(0),
                              thrust::make_counting_iterator<size_t>(image.height_ * image.width_ * 3),
                              render_image, func);
        } else if (image.num_of_channels_ == 3 &&
                   image.bytes_per_channel_ == 2) {
            // image with RGB channels, each channel is a 16-bit integer
            copy_int16_rgb_image_functor func(thrust::raw_pointer_cast(image.data_.data()));
            thrust::transform(thrust::make_counting_iterator<size_t>(0),
                              thrust::make_counting_iterator<size_t>(image.height_ * image.width_ * 3),
                              render_image, func);
        } else if (image.num_of_channels_ == 1 &&
                   image.bytes_per_channel_ == 2) {
            // depth image, one channel of 16-bit integer
            const int max_depth = option.image_max_depth_;
            copy_depth_image_functor func(thrust::raw_pointer_cast(image.data_.data()), max_depth);
            thrust::transform(thrust::make_counting_iterator<size_t>(0),
                              thrust::make_counting_iterator<size_t>(image.height_ * image.width_ * 3),
                              render_image, func);
        }
    }

    draw_arrays_mode_ = GL_TRIANGLES;
    draw_arrays_size_ = 6;
    return true;
}

size_t ImageShaderForImage::GetDataSize(const geometry::Geometry &geometry) const {
    return ((const geometry::Image &)geometry).data_.size();
}

size_t ImageShaderForImage::GetDataHeight(const geometry::Geometry &geometry) const {
    return ((const geometry::Image &)geometry).height_;
}

size_t ImageShaderForImage::GetDataWidth(const geometry::Geometry &geometry) const {
    return ((const geometry::Image &)geometry).width_;
}