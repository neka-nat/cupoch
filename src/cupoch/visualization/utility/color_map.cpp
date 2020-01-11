#include "cupoch/visualization/utility/color_map.h"

namespace cupoch {

namespace {
using namespace visualization;
static ColorMap::ColorMapOption global_option = ColorMap::ColorMapOption::Jet;
}  // unnamed namespace

namespace visualization {

ColorMap::ColorMapOption GetGlobalColorMapOption() {
    return global_option;
}

void SetGlobalColorMapOption(ColorMap::ColorMapOption option) {
    global_option = option;
}

}  // namespace visualization
}  // namespace cupoch