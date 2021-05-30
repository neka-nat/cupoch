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
#include "cupoch/utility/console.h"

#include <cstdio>
#include <ctime>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cupoch {
namespace utility {

std::string GetCurrentTimeStamp() {
    std::time_t t = std::time(nullptr);
    struct tm *timeinfo;
    timeinfo = localtime(&t);
    std::string fmt = "{:%Y-%m-%d-%H-%M-%S}\a";
    std::string buffer;
    buffer.resize(fmt.size() + 1);
    int len = strftime(&buffer[0], buffer.size(), fmt.c_str(), timeinfo);
    while (len == 0) {
        buffer.resize(buffer.size() * 2);
        len = strftime(&buffer[0], buffer.size(), fmt.c_str(), timeinfo);
    }
    buffer.resize(len - 1);
    return buffer;
}

}  // namespace utility
}  // namespace cupoch