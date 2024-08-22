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

#include <spdlog/spdlog.h>

#define DEFAULT_IO_BUFFER_SIZE 1024

namespace cupoch {
namespace utility {

enum class VerbosityLevel {
    Off = SPDLOG_LEVEL_OFF,
    Fatal = SPDLOG_LEVEL_CRITICAL,
    Error = SPDLOG_LEVEL_ERROR,
    Warning = SPDLOG_LEVEL_WARN,
    Info = SPDLOG_LEVEL_INFO,
    Debug = SPDLOG_LEVEL_DEBUG,
    trace = SPDLOG_LEVEL_TRACE,
};

inline void SetVerbosityLevel(VerbosityLevel level) {
    spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
}

inline VerbosityLevel GetVerbosityLevel() {
    return static_cast<VerbosityLevel>(spdlog::get_level());
}

template <typename... Args>
inline void LogFatal(const char *format, Args &&... args) {
    spdlog::critical(format, std::forward<Args>(args)...);
}

template <typename... Args>
inline void LogError(const char *format, Args &&... args) {
    spdlog::error(format, std::forward<Args>(args)...);
}

template <typename... Args>
inline void LogWarning(const char *format, Args &&... args) {
    spdlog::warn(format, std::forward<Args>(args)...);
}

template <typename... Args>
inline void LogInfo(const char *format, Args &&... args) {
    spdlog::info(format, std::forward<Args>(args)...);
}

template <typename... Args>
inline void LogDebug(const char *format, Args &&... args) {
    spdlog::debug(format, std::forward<Args>(args)...);
}

class ConsoleProgressBar {
public:
    ConsoleProgressBar(size_t expected_count,
                       const std::string &progress_info,
                       bool active = false) {
        reset(expected_count, progress_info, active);
    }

    void reset(size_t expected_count,
               const std::string &progress_info,
               bool active) {
        expected_count_ = expected_count;
        current_count_ = -1;
        progress_info_ = progress_info;
        progress_pixel_ = 0;
        active_ = active;
        operator++();
    }

    ConsoleProgressBar &operator++() {
        current_count_++;
        if (!active_) {
            return *this;
        }
        if (current_count_ >= expected_count_) {
            printf("%s[%s] 100%%\n", progress_info_.c_str(),
                   std::string(resolution_, '=').c_str());
        } else {
            size_t new_progress_pixel =
                    int(current_count_ * resolution_ / expected_count_);
            if (new_progress_pixel > progress_pixel_) {
                progress_pixel_ = new_progress_pixel;
                int percent = int(current_count_ * 100 / expected_count_);
                printf("%s[%s>%s] %d%%\r", progress_info_.c_str(),
                       std::string(progress_pixel_, '=').c_str(),
                       std::string(resolution_ - 1 - progress_pixel_, ' ')
                               .c_str(),
                       percent);
                fflush(stdout);
            }
        }
        return *this;
    }

private:
    const size_t resolution_ = 40;
    size_t expected_count_;
    int current_count_;
    std::string progress_info_;
    size_t progress_pixel_;
    bool active_;
};

std::string GetCurrentTimeStamp();

}  // namespace utility
}  // namespace cupoch