#include "cupoch/visualization/visualizer/visualizer.h"

using namespace cupoch;
using namespace cupoch::visualization;

void Visualizer::WindowRefreshCallback(GLFWwindow *window) {
    if (is_redraw_required_) {
        Render();
        is_redraw_required_ = false;
    }
}