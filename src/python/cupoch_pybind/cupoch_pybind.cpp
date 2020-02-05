#include "cupoch_pybind/cupoch_pybind.h"
#include "cupoch_pybind/camera/camera.h"
#include "cupoch_pybind/geometry/geometry.h"
#include "cupoch_pybind/io/io.h"
#include "cupoch_pybind/odometry/odometry.h"
#include "cupoch_pybind/registration/registration.h"
#include "cupoch_pybind/utility/utility.h"
#include "cupoch_pybind/visualization/visualization.h"

PYBIND11_MODULE(cupoch, m) {
    m.doc() = "CUDA-based 3D data processing library";

    pybind_utility(m);
    pybind_camera(m);
    pybind_geometry(m);
    pybind_io(m);
    pybind_registration(m);
    pybind_odometry(m);
    pybind_visualization(m);
}