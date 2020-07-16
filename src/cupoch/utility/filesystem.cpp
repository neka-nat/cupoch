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