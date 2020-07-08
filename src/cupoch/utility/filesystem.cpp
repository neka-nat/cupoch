#include "cupoch/utility/filesystem.h"

#include <algorithm>
#ifdef WINDOWS
#include <direct.h>
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#else
#include <dirent.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cupoch {
namespace utility {
namespace filesystem {

std::string GetFileExtensionInLowerCase(const std::string &filename) {
    size_t dot_pos = filename.find_last_of(".");
    if (dot_pos >= filename.length()) return "";

    if (filename.find_first_of("/\\", dot_pos) != std::string::npos) return "";

    std::string filename_ext = filename.substr(dot_pos + 1);

    std::transform(filename_ext.begin(), filename_ext.end(),
                   filename_ext.begin(), ::tolower);

    return filename_ext;
}

std::string GetFileNameWithoutExtension(const std::string &filename) {
    size_t dot_pos = filename.find_last_of(".");

    return filename.substr(0, dot_pos);
}

std::string GetFileNameWithoutDirectory(const std::string &filename) {
    size_t slash_pos = filename.find_last_of("/\\");
    if (slash_pos == std::string::npos) {
        return filename;
    } else {
        return filename.substr(slash_pos + 1);
    }
}

std::string GetFileParentDirectory(const std::string &filename) {
    size_t slash_pos = filename.find_last_of("/\\");
    if (slash_pos == std::string::npos) {
        return "";
    } else {
        return filename.substr(0, slash_pos + 1);
    }
}

std::string GetWorkingDirectory() {
    char buff[PATH_MAX + 1];
    getcwd(buff, PATH_MAX + 1);
    return std::string(buff);
}

bool ChangeWorkingDirectory(const std::string &directory) {
    return (chdir(directory.c_str()) == 0);
}

FILE *FOpen(const std::string &filename, const std::string &mode) {
    FILE *fp;
#ifndef _WIN32
    fp = fopen(filename.c_str(), mode.c_str());
#else
    std::wstring filename_w;
    filename_w.resize(filename.size());
    int newSize = MultiByteToWideChar(
            CP_UTF8, 0, filename.c_str(), filename.length(),
            const_cast<wchar_t *>(filename_w.c_str()), filename.length());
    filename_w.resize(newSize);
    std::wstring mode_w(mode.begin(), mode.end());
    fp = _wfopen(filename_w.c_str(), mode_w.c_str());
#endif
    return fp;
}

}  // namespace filesystem
}  // namespace utility
}  // namespace cupoch