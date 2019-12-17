find_package(PkgConfig QUIET)
find_package(pybind11  QUIET)
find_package(OpenGL    QUIET)

if (PKGCONFIG_FOUND)
    pkg_search_module(EIGEN3          eigen3>=3.2.7   QUIET)
    pkg_search_module(GLFW            glfw3           QUIET)
    pkg_search_module(GLEW            glew            QUIET)
endif (PKGCONFIG_FOUND)

macro(INSTALL_HEADERS source)
    install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
            DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/3rdparty"
            PATTERN     "*.c"           EXCLUDE
            PATTERN     "*.cmake"       EXCLUDE
            PATTERN     "*.cpp"         EXCLUDE
            PATTERN     "*.in"          EXCLUDE
            PATTERN     "*.m"           EXCLUDE
            PATTERN     "*.txt"         EXCLUDE
            PATTERN     ".gitignore"    EXCLUDE)
endmacro()

if (BUILD_PYBIND11)
    add_subdirectory(pybind11)
endif ()

if (BUILD_UNIT_TESTS)
    message(STATUS "Building googletest from source")
    include_directories("googletest/googletest/include")
    include_directories("googletest/googletest/")
    set(googletest_INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googlemock/include
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googlemock/)
    set(googletest_SOURCE
        googletest/googletest/src/gtest-all.cc
        googletest/googlemock/src/gmock-all.cc)
    include_directories(${googletest_INCLUDE_DIRS})
    add_library(googletest STATIC ${googletest_SOURCE})
    if (NOT BUILD_SHARED_LIBS)
      install(TARGETS googletest
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    endif()
    set(googletest_LIBRARIES googletest)
endif()

if (BUILD_EIGEN3)
    set(EIGEN3_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/third_party/eigen")
elseif (EIGEN3_FOUND)
    message(STATUS "Using installed EIGEN3 ${EIGEN3_VERSION}")
else ()
    message(STATUS "Unable to find EIGEN3 installed in the system")
    message(STATUS "Building EIGEN3 from source")
    set(BUILD_EIGEN3 ON)
endif ()

set(flann_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/flann/src/cpp)

file(GLOB_RECURSE C_SOURCES flann/src/cpp/flann/flann.cpp)
file(GLOB_RECURSE CPP_SOURCES flann/src/cpp/flann/flann_cpp.cpp)
file(GLOB_RECURSE CU_SOURCES flann/src/cpp/flann/*.cu)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DFLANN_USE_CUDA")
cuda_add_library(flann_cuda_s STATIC ${CU_SOURCES})
target_include_directories(flann_cuda_s PUBLIC ${flann_INCLUDE_DIRS})

# GLEW
if (BUILD_GLEW)
    message(STATUS "Building GLEW from source (BUILD_GLEW=ON)")
elseif (GLEW_FOUND)
    message(STATUS "Using installed GLEW ${GLEW_VERSION}")
else ()
    message(STATUS "Unable to find GLEW installed in the system")
    message(STATUS "Building GLEW from source")
    set(BUILD_GLEW ON)
endif ()

if (BUILD_GLEW)
    add_subdirectory(glew)
    INSTALL_HEADERS(glew)
endif ()

# GLFW
if (BUILD_GLFW)
    message(STATUS "Building GLFW from source (BUILD_GLFW=ON)")
elseif (GLFW_FOUND AND OPENGL_FOUND)
    message(STATUS "Using installed GLFW ${GLFW_VERSION}")
    if (APPLE)
        find_library(COCOA_FRAMEWORK Cocoa)
        find_library(IOKIT_FRAMEWORK IOKit)
        find_library(CORE_FOUNDATION_FRAMEWORK CoreFoundation)
        find_library(CORE_VIDEO_FRAMEWORK CoreVideo)
        mark_as_advanced(COCOA_FRAMEWORK
                         IOKIT_FRAMEWORK
                         CORE_FOUNDATION_FRAMEWORK
                         CORE_VIDEO_FRAMEWORK)
        list(APPEND GLFW_LIBRARIES "${COCOA_FRAMEWORK}"
                                    "${OPENGL_gl_LIBRARY}"
                                    "${IOKIT_FRAMEWORK}"
                                    "${CORE_FOUNDATION_FRAMEWORK}"
                                    "${CORE_VIDEO_FRAMEWORK}")
    else ()
        list(APPEND GLFW_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR})
        list(APPEND GLFW_LIBRARIES    ${OPENGL_gl_LIBRARY})
    endif ()
else ()
    message(STATUS "Unable to find GLFW installed in the system")
    message(STATUS "Building GLFW from source")
    set(BUILD_GLFW ON)
    # Notify PARENT_SCOPE that GLFW was built, to deal with glfw(3) naming issue
    set(BUILD_GLFW ON PARENT_SCOPE)
endif ()

if (BUILD_GLFW)
    add_subdirectory(GLFW)
    INSTALL_HEADERS(GLFW)
    list(APPEND GLFW_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR})
    list(APPEND GLFW_LIBRARIES    ${OPENGL_gl_LIBRARY})
endif ()

cuda_add_library(fmt fmt/src/format.cc fmt/src/posix.cc)
target_include_directories(fmt PUBLIC fmt/include)
set(fmt_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/fmt/include)

# liblzf
file(GLOB LIBLZF_SOURCE_FILES "liblzf/*.c")
add_library(liblzf ${LIBLZF_SOURCE_FILES})
target_include_directories(liblzf PUBLIC liblzf)
set(liblzf_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/liblzf)

# rply
add_library(rply rply/rply.c)
set(rply_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/rply)

# dlpack
set(dlpack_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/dlpack/include)

# rmm
add_library(rmm SHARED rmm/src/rmm.cpp
            rmm/src/memory_manager.cpp
            cnmem/src/cnmem.cpp
            rmm/src/mr/default_memory_resource.cpp)
target_include_directories(rmm PUBLIC cnmem/include rmm/include)
target_link_libraries(rmm ${CUDA_LIBRAIES})
set(rmm_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/rmm/include)

# imgui
add_library(imgui imgui/imgui.cpp
            imgui/imgui_draw.cpp
            imgui/imgui_widgets.cpp)
set(imgui_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

list(APPEND 3RDPARTY_INCLUDE_DIRS
     ${EIGEN3_INCLUDE_DIRS}
     ${flann_INCLUDE_DIRS}
     ${rply_INCLUDE_DIRS}
     ${fmt_INCLUDE_DIRS}
     ${liblzf_INCLUDE_DIRS}
     ${dlpack_INCLUDE_DIRS}
     ${rmm_INCLUDE_DIRS}
     ${imgui_INCLUDE_DIRS}
)

set(3RDPARTY_INCLUDE_DIRS ${3RDPARTY_INCLUDE_DIRS} PARENT_SCOPE)