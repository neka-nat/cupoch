#include "cupoch/io/class_io/image_io.h"
#include "cupoch/geometry/image.h"

#include <unordered_map>

#include "cupoch/utility/console.h"
#include "cupoch/utility/filesystem.h"

namespace cupoch {

namespace {
using namespace io;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::Image &)>>
        file_extension_to_image_read_function{
                {"png", ReadImageFromPNG},
                {"jpg", ReadImageFromJPG},
                {"jpeg", ReadImageFromJPG},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, const geometry::Image &, int)>>
        file_extension_to_image_write_function{
                {"png", WriteImageToPNG},
                {"jpg", WriteImageToJPG},
                {"jpeg", WriteImageToJPG},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, const HostImage &, int)>>
        file_extension_to_host_image_write_function{
                {"png", WriteHostImageToPNG},
        };

}  // unnamed namespace

namespace io {

std::shared_ptr<geometry::Image> CreateImageFromFile(
        const std::string &filename) {
    auto image = std::make_shared<geometry::Image>();
    ReadImage(filename, *image);
    return image;
}

bool ReadImage(const std::string &filename, geometry::Image &image) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::Image failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_image_read_function.find(filename_ext);
    if (map_itr == file_extension_to_image_read_function.end()) {
        utility::LogWarning(
                "Read geometry::Image failed: unknown file extension.");
        return false;
    }
    return map_itr->second(filename, image);
}

bool WriteImage(const std::string &filename,
                const geometry::Image &image,
                int quality /* = 90*/) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::Image failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_image_write_function.find(filename_ext);
    if (map_itr == file_extension_to_image_write_function.end()) {
        utility::LogWarning(
                "Write geometry::Image failed: unknown file extension.");
        return false;
    }
    return map_itr->second(filename, image, quality);
}

bool WriteImage(const std::string &filename,
                const HostImage &image,
                int quality /* = 90*/) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::Image failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_host_image_write_function.find(filename_ext);
    if (map_itr == file_extension_to_host_image_write_function.end()) {
        utility::LogWarning(
                "Write geometry::Image failed: unknown file extension.");
        return false;
    }
    return map_itr->second(filename, image, quality);
}

}  // namespace io
}  // namespace cupoch