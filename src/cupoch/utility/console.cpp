#include "cupoch/utility/console.h"

#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#endif

namespace cupoch {
namespace utility {

void Logger::ChangeConsoleColor(TextColor text_color, int highlight_text) {
#ifdef _WIN32
    const WORD EMPHASIS_MASK[2] = {0, FOREGROUND_INTENSITY};
    const WORD COLOR_MASK[8] = {
            0,
            FOREGROUND_RED,
            FOREGROUND_GREEN,
            FOREGROUND_GREEN | FOREGROUND_RED,
            FOREGROUND_BLUE,
            FOREGROUND_RED | FOREGROUND_BLUE,
            FOREGROUND_GREEN | FOREGROUND_BLUE,
            FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED};
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(
            h, EMPHASIS_MASK[highlight_text] | COLOR_MASK[(int)text_color]);
#else
    printf("%c[%d;%dm", 0x1B, highlight_text, (int)text_color + 30);
#endif
}

void Logger::ResetConsoleColor() {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(
            h, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED);
#else
    printf("%c[0;m", 0x1B);
#endif
}

}  // namespace utility
}  // namespace cupoch