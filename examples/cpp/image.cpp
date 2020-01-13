#include "cupoch/cupoch.h"

int main(int argc, char **argv) {
    using namespace cupoch;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    if (argc < 2) {utility::LogInfo("Need an argument of image file name."); return 0;}
    auto color_image_8bit = io::CreateImageFromFile(argv[1]);
    utility::LogDebug("RGB image size : {:d} x {:d}",
                      color_image_8bit->width_, color_image_8bit->height_);
    io::WriteImage("copy.png",
                   *color_image_8bit);
}