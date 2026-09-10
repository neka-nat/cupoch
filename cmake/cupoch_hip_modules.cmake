# Copyright (c) 2026 Advanced Micro Devices, Inc.
# ROCm/HIP port author: Jeff Daily <jeff.daily@amd.com>
#
# Full ROCm/HIP module assembly. Builds every cupoch module that is portable to
# HIP. The two modules with hard external coupling are decided here, not by a
# user flag:
#   - visualization: needs an OpenGL context + GL<->HIP interop. ROCm ships the
#     hipGraphics* GL-interop API; visualization is attempted when OpenGL/GLEW/
#     GLFW are available and disabled automatically otherwise.
#   - imageproc: a thin wrapper over the CUDA-only libSGM stereo library, which
#     is not ported to HIP. Disabled automatically.
# Everything else (collision, integration, io, kinematics, kinfu, odometry,
# planning, registration + the core utility/camera/knn/geometry) builds on HIP.

configure_file("${PROJECT_SOURCE_DIR}/src/cupoch/cupoch_config.h.in"
               "${PROJECT_SOURCE_DIR}/src/cupoch/cupoch_config.h")

# Assembles the dependency graph and sets CUPOCH_HIP_BUILD_VISUALIZATION based
# on OpenGL availability (the GL renderer is attempted when an OpenGL context is
# present; ROCm provides the hipGraphics* GL<->HIP interop it needs).
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cupoch_hip_3rdparty.cmake)

include_directories(SYSTEM ${3RDPARTY_INCLUDE_DIRS})
include_directories(src)

# Core + compute modules (portable on HIP).
add_subdirectory(src/cupoch/utility)
add_subdirectory(src/cupoch/camera)
add_subdirectory(src/cupoch/knn)
add_subdirectory(src/cupoch/geometry)
add_subdirectory(src/cupoch/collision)
add_subdirectory(src/cupoch/integration)
add_subdirectory(src/cupoch/io)
add_subdirectory(src/cupoch/odometry)
add_subdirectory(src/cupoch/planning)
add_subdirectory(src/cupoch/registration)
add_subdirectory(src/cupoch/kinematics)
add_subdirectory(src/cupoch/kinfu)

# imageproc is libSGM-only (CUDA stereo, not ported to HIP); skipped on HIP.

if (CUPOCH_HIP_BUILD_VISUALIZATION)
    # The visualization module generates shader.h from GLSL via encode_shader.py.
    if (NOT PYTHON_EXECUTABLE)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if (Python3_Interpreter_FOUND)
            set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
        endif ()
    endif ()
    add_subdirectory(src/cupoch/visualization)
endif ()

if (BUILD_PYTHON_MODULE)
    # pybind11 provides PYBIND11_INCLUDE_DIR / PYTHON_* the python target needs.
    if (BUILD_PYBIND11)
        # See third_party/CMakeLists.txt for why classic mode is forced.
        set(PYBIND11_FINDPYTHON OFF)
        add_subdirectory(third_party/pybind11 ${CMAKE_BINARY_DIR}/pybind11)
    endif ()
    add_subdirectory(src/python)
endif ()
