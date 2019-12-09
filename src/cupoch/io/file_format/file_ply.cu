#include <rply.h>
#include "cupoch/io/class_io/pointcloud_io.h"
#include "cupoch/utility/console.h"

namespace cupoch {

namespace {
using namespace io;

namespace ply_pointcloud_reader {

struct PLYReaderState {
    utility::ConsoleProgressBar *progress_bar;
    HostPointCloud *pointcloud_ptr;
    long vertex_index;
    long vertex_num;
    long normal_index;
    long normal_num;
    long color_index;
    long color_num;
};

int ReadVertexCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->vertex_index >= state_ptr->vertex_num) {
        return 0;  // some sanity check
    }

    float value = ply_get_argument_value(argument);
    state_ptr->pointcloud_ptr->points_[state_ptr->vertex_index](index) = value;
    if (index == 2) {  // reading 'z'
        state_ptr->vertex_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

int ReadNormalCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->normal_index >= state_ptr->normal_num) {
        return 0;
    }

    float value = ply_get_argument_value(argument);
    state_ptr->pointcloud_ptr->normals_[state_ptr->normal_index](index) = value;
    if (index == 2) {  // reading 'nz'
        state_ptr->normal_index++;
    }
    return 1;
}

int ReadColorCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->color_index >= state_ptr->color_num) {
        return 0;
    }

    float value = ply_get_argument_value(argument);
    state_ptr->pointcloud_ptr->colors_[state_ptr->color_index](index) =
            value / 255.0;
    if (index == 2) {  // reading 'blue'
        state_ptr->color_index++;
    }
    return 1;
}

}  // namespace ply_pointcloud_reader
}

namespace io {

bool ReadPointCloudFromPLY(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           bool print_progress) {
    using namespace ply_pointcloud_reader;

    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Read PLY failed: unable to open file: %s",
                            filename.c_str());
        return false;
    }
    if (!ply_read_header(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to parse header.");
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;
    HostPointCloud host_pc;
    state.pointcloud_ptr = &host_pc;
    state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x",
                                       ReadVertexCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state, 2);

    state.normal_num = ply_set_read_cb(ply_file, "vertex", "nx",
                                       ReadNormalCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "ny", ReadNormalCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "nz", ReadNormalCallback, &state, 2);

    state.color_num = ply_set_read_cb(ply_file, "vertex", "red",
                                      ReadColorCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "green", ReadColorCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "blue", ReadColorCallback, &state, 2);

    if (state.vertex_num <= 0) {
        utility::LogWarning("Read PLY failed: number of vertex <= 0.");
        ply_close(ply_file);
        return false;
    }

    state.vertex_index = 0;
    state.normal_index = 0;
    state.color_index = 0;

    host_pc.Clear();
    host_pc.points_.resize(state.vertex_num);
    host_pc.normals_.resize(state.normal_num);
    host_pc.colors_.resize(state.color_num);

    utility::ConsoleProgressBar progress_bar(state.vertex_num + 1,
                                             "Reading PLY: ", print_progress);
    state.progress_bar = &progress_bar;

    if (!ply_read(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to read file: {}",
                            filename);
        ply_close(ply_file);
        return false;
    }

    ply_close(ply_file);
    ++progress_bar;
    host_pc.ToDevice(pointcloud);
    return true;
}

bool WritePointCloudToPLY(const std::string &filename,
                          const geometry::PointCloud &pointcloud,
                          bool write_ascii /* = false*/,
                          bool compressed /* = false*/,
                          bool print_progress) {
    if (pointcloud.IsEmpty()) {
        utility::LogWarning("Write PLY failed: point cloud has 0 points.");
        return false;
    }

    p_ply ply_file = ply_create(filename.c_str(),
                                write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN,
                                NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Write PLY failed: unable to open file: {}",
                            filename);
        return false;
    }
    ply_add_comment(ply_file, "Created by Cupoch");
    ply_add_element(ply_file, "vertex",
                    static_cast<long>(pointcloud.points_.size()));
    ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    if (pointcloud.HasNormals()) {
        ply_add_property(ply_file, "nx", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "ny", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "nz", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    }
    if (pointcloud.HasColors()) {
        ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    }
    if (!ply_write_header(ply_file)) {
        utility::LogWarning("Write PLY failed: unable to write header.");
        ply_close(ply_file);
        return false;
    }

    utility::ConsoleProgressBar progress_bar(
            static_cast<size_t>(pointcloud.points_.size()),
            "Writing PLY: ", print_progress);

    bool printed_color_warning = false;
    HostPointCloud host_pc;
    host_pc.FromDevice(pointcloud);
    for (size_t i = 0; i < pointcloud.points_.size(); i++) {
        const Eigen::Vector3f &point = host_pc.points_[i];
        ply_write(ply_file, point(0));
        ply_write(ply_file, point(1));
        ply_write(ply_file, point(2));
        if (pointcloud.HasNormals()) {
            const Eigen::Vector3f &normal = host_pc.normals_[i];
            ply_write(ply_file, normal(0));
            ply_write(ply_file, normal(1));
            ply_write(ply_file, normal(2));
        }
        if (pointcloud.HasColors()) {
            const Eigen::Vector3f &color = host_pc.colors_[i];
            if (!printed_color_warning &&
                (color(0) < 0 || color(0) > 1 || color(1) < 0 || color(1) > 1 ||
                 color(2) < 0 || color(2) > 1)) {
                utility::LogWarning(
                        "Write Ply clamped color value to valid range");
                printed_color_warning = true;
            }
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(0) * 255.0)));
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(1) * 255.0)));
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(2) * 255.0)));
        }
        ++progress_bar;
    }

    ply_close(ply_file);
    return true;
}

}
}