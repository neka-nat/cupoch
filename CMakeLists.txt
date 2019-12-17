cmake_minimum_required(VERSION 3.0.0)

if (NOT CMAKE_VERSION VERSION_LESS "3.1")
    cmake_policy(SET CMP0054 NEW)
endif()

project(cupoch)

option(BUILD_UNIT_TESTS          "Build the Cupoch unit tests"              ON)
option(BUILD_EIGEN3              "Use the Eigen3 that comes with Cupoch"    ON)
option(BUILD_GLEW                "Build glew from source"                   OFF)
option(BUILD_GLFW                "Build glfw from source"                   OFF)
option(BUILD_PYBIND11            "Build pybind11 from source"               ON)
option(BUILD_PYTHON_MODULE       "Build the python module"                  ON)
option(STATIC_WINDOWS_RUNTIME    "Use static (MT/MTd) Windows runtime"      OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

if (WIN32)
    # can't hide the unit testing option on Windows only
    # as a precaution: disable unit testing on Windows regardless of user input
    message(STATUS "Disable unit tests since this feature is not fully supported on Windows.")
    set(BUILD_UNIT_TESTS OFF)
    add_definitions(-DWINDOWS)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)		# suppress C4996 warning
    add_definitions(-DGLEW_STATIC)		# enable GLEW for windows
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    message(STATUS "Compiling on Windows")
    if (MSVC)
        message(STATUS "Compiling with MSVC")
        add_definitions(-DNOMINMAX)
        add_definitions(-D_USE_MATH_DEFINES)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D _ENABLE_EXTENDED_ALIGNED_STORAGE")
        # Multi-thread compile, two ways to enable
        # Option 1, at build time: cmake --build . --parallel %NUMBER_OF_PROCESSORS%
        # https://stackoverflow.com/questions/36633074/set-the-number-of-threads-in-a-cmake-build
        # OPtion 2, at configure time: add /MP flag, no need to use Option 1
        # https://docs.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes?view=vs-2019
        add_definitions("/MP")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /MP")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MP")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MP")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MP")
    endif (MSVC)

    if (STATIC_WINDOWS_RUNTIME)
        # by default, "/MD" and "/MDd" is set by CMake automatically
        string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
        string(REPLACE "/MDd" "/MTd" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
        string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
        string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    else ()
        # handles the case when re-running cmake with STATIC_WINDOWS_RUNTIME=OFF
        string(REPLACE "/MT" "/MD" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
        string(REPLACE "/MTd" "/MDd" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
        string(REPLACE "/MT" "/MD" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
        string(REPLACE "/MTd" "/MDd" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    endif ()
    message(STATUS "CMAKE_C_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "CMAKE_C_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
    message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
elseif (CYGWIN)
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    message(STATUS "Compiling on Cygwin")
    # enable c++11
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    add_definitions(-DCYGWIN)
elseif (APPLE)
    add_definitions(-DUNIX)
    # enable c++11
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    add_compile_options(-Wno-deprecated-declarations)
    if (NOT BUILD_SHARED_LIBS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    endif (NOT BUILD_SHARED_LIBS)
    # In Release build -O3 will be added automatically by CMake
    # We still enable -O3 at Debug build to optimize performance
    if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        add_definitions(-O3)
    endif()
elseif (UNIX)
    add_definitions(-DUNIX)
    # enable c++11
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC")
    add_compile_options(-Wno-deprecated-declarations)
    add_compile_options(-Wno-unused-result)
    # In Release build -O3 will be added automatically by CMake
    # We still enable -O3 at debug build to optimize performance
    if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        add_definitions(-O3)
    endif()
    # disable BUILD_LIBREALSENSE since it is not fully supported on Linux
    message(STATUS "Compiling on Unix")
endif ()

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_ARCH -gencode arch=compute_30,code=sm_30
              -gencode arch=compute_35,code=sm_35
              -gencode arch=compute_37,code=sm_37
              -gencode arch=compute_50,code=sm_50
              -gencode arch=compute_52,code=sm_52
              -gencode arch=compute_60,code=sm_60
              -gencode arch=compute_61,code=sm_61
              -gencode arch=compute_70,code=sm_70)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --default-stream per-thread
    -Xcudafe "--diag_suppress=integer_sign_change"
    -Xcudafe "--diag_suppress=partial_override"
    -Xcudafe "--diag_suppress=virtual_function_decl_hidden"
    ${CUDA_ARCH})

add_subdirectory(third_party)

include_directories(
    SYSTEM
    ${3RDPARTY_INCLUDE_DIRS}
)

include_directories(src)
add_subdirectory(src)
add_subdirectory(examples)