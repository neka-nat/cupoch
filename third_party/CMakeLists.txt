find_package(PkgConfig QUIET)
find_package(pybind11  QUIET)
find_package(OpenGL    QUIET)

include(GenerateExportHeader)

if (PKGCONFIG_FOUND)
    pkg_search_module(EIGEN3          eigen3>=3.2.7   QUIET)
    pkg_search_module(GLFW            glfw3           QUIET)
    pkg_search_module(GLEW            glew            QUIET)
    pkg_search_module(JSONCPP         jsoncpp>=1.7.0  QUIET)
    pkg_search_module(PNG             libpng>=1.6.0   QUIET)
    pkg_search_module(JPEG_TURBO      libturbojpeg    QUIET)
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

set(flann_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE CU_SOURCES flann/*.cu)
cuda_add_library(flann_cuda_s STATIC ${CU_SOURCES})
target_include_directories(flann_cuda_s PRIVATE ${flann_INCLUDE_DIRS} rmm/include spdlog/include)
target_compile_definitions(flann_cuda_s PRIVATE -DFLANN_USE_CUDA)
set(FLANN_LIBRARIES flann_cuda_s)

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

set(SPDLOG_MASTER_PROJECT OFF)
add_subdirectory(spdlog)
set(spdlog_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/spdlog/include)

# JSONCPP
if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU"))
    # It is a known bug that g++ on OSX only supports libstdc++
    # The jsoncpp installed by brew by default is compiled with libc++
    # If the two libraries are linked together, they break the build
    # https://github.com/open-source-parsers/jsoncpp/issues/597
    # https://github.com/IntelVCL/Open3D/issues/9
    message(STATUS "Building JSONCPP from source, without installing headers")
    add_subdirectory(jsoncpp)
    set(BUILD_JSONCPP ON)
else ()
    if (BUILD_JSONCPP)
        message(STATUS "Building JSONCPP from source (BUILD_JSONCPP=ON)")
    elseif (JSONCPP_FOUND)
        message(STATUS "Using installed JSONCPP ${JSONCPP_VERSION}")
    else ()
        message(STATUS "Unable to find JSONCPP installed in the system")
        message(STATUS "Building JSONCPP from source")
        set(BUILD_JSONCPP ON)
    endif ()

    if (BUILD_JSONCPP)
        add_subdirectory(jsoncpp)
    endif ()
endif ()

# liblzf
file(GLOB LIBLZF_SOURCE_FILES "liblzf/*.c")
add_library(liblzf ${LIBLZF_SOURCE_FILES})
target_include_directories(liblzf PUBLIC liblzf)
set(liblzf_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/liblzf)
set(liblzf_LIBRARIES liblzf)

# tritriintersect
set(tomasakeninemoeller_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/tomasakeninemoeller")

# PNG
if (BUILD_PNG)
    message(STATUS "Building LIBPNG from source (BUILD_PNG=ON)")
elseif (PNG_FOUND)
    message(STATUS "Using installed LIBPNG ${PNG_VERSION}")
else ()
    message(STATUS "Unable to find libpng installed in the system")
    message(STATUS "Building libpng from source")
    set(BUILD_PNG on)
endif ()

if (BUILD_PNG)
    add_subdirectory(zlib)
    add_subdirectory(libpng)
    list(APPEND PNG_LIBRARIES zlib)
endif ()

# JPEG
if (BUILD_JPEG)
    message(STATUS "Building libjpeg-turbo from source (BUILD_JPEG=ON)")
elseif (JPEG_TURBO_FOUND)
    message(STATUS "Using installed libjpeg-turbo ${JPEG_TUROB_VERSION}")
    message(STATUS ${JPEG_TURBO_INCLUDE_DIRS})
    message(STATUS ${JPEG_TURBO_LIBRARIES})
else ()
    message(STATUS "Unable to find libjpeg-turbo installed in the system")
    message(STATUS "Building libjpeg-turbo from source")
    set(BUILD_JPEG ON)
endif ()

if (BUILD_JPEG)
    message(STATUS "Building libjpeg-turbo from source")
    include(libjpeg-turbo/libjpeg-turbo.cmake)
    set(JPEG_TURBO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo/libjpeg-turbo
        ${CMAKE_CURRENT_BINARY_DIR}/turbojpeg/src/ext_turbojpeg-build)
    set(JPEG_TURBO_LIBRARIES turbojpeg)
endif()

# stdgpu
set(STDGPU_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(STDGPU_BUILD_TESTS OFF CACHE INTERNAL "")
set(STDGPU_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
set(STDGPU_SETUP_COMPILER_FLAGS OFF CACHE INTERNAL "")
add_subdirectory(stdgpu)
target_compile_options(stdgpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fPIC>)
target_compile_options(stdgpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fPIC>)
set(stdgpu_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/stdgpu/src ${CMAKE_CURRENT_BINARY_DIR}/stdgpu/src/stdgpu/include)
set(stdgpu_LIBRARIES stdgpu)

# tinyobjloader
message(STATUS "Building tinyobjloader from source")
add_library(tinyobjloader STATIC tinyobjloader/tiny_obj_loader.cc)
target_include_directories(tinyobjloader PUBLIC "tinyobjloader/")
if (NOT BUILD_SHARED_LIBS)
  install(TARGETS tinyobjloader
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
set(tinyobjloader_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/tinyobjloader")
set(tinyobjloader_LIBRARIES tinyobjloader)

# rply
add_library(rply rply/rply.c)
set(rply_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/rply)
set(rply_LIBRARIES rply)

# dlpack
set(dlpack_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/dlpack/include)

# rmm
set(rmm_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/rmm/include)

# imgui
add_library(imgui STATIC imgui/imgui.cpp
            imgui/imgui_draw.cpp
            imgui/imgui_tables.cpp
            imgui/imgui_widgets.cpp
            imgui/backends/imgui_impl_glfw.cpp
            imgui/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/imgui
                           ${GLFW_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})
set(imgui_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
set(imgui_LIBRARIES imgui)

# lbvh
set(lbvh_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lbvh)

# lbvh_index
set(lbvh_index_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lbvh_index)

# urdfdom
set(CONSOLE_BRIDGE_MAJOR_VERSION 1)
set(CONSOLE_BRIDGE_MINOR_VERSION 0)
set(CONSOLE_BRIDGE_PATCH_VERSION 1)
add_library(console_bridge urdfdom/urdf_parser/src/console.cpp)
set_target_properties(console_bridge PROPERTIES SOVERSION
               ${CONSOLE_BRIDGE_MAJOR_VERSION}.${CONSOLE_BRIDGE_MINOR_VERSION})
target_include_directories(console_bridge PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/urdfdom/urdf_parser/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
	$<INSTALL_INTERFACE:include>)
generate_export_header(console_bridge
    EXPORT_MACRO_NAME CONSOLE_BRIDGE_DLLAPI)

file(GLOB_RECURSE URDFDOM_SOURCES urdfdom/urdf_parser/src/*.cpp)
list(REMOVE_ITEM URDFDOM_SOURCES "urdfdom/urdf_parser/src/console.cpp")
add_library(urdfdom STATIC ${URDFDOM_SOURCES})
target_include_directories(urdfdom PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/urdfdom/urdf_parser/include
                           ${CMAKE_CURRENT_SOURCE_DIR}/urdfdom/urdf_parser/include/tinyxml)
target_link_libraries(urdfdom console_bridge)
set(urdfdom_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/urdfdom/urdf_parser/include
    ${CMAKE_CURRENT_SOURCE_DIR}/urdfdom/urdf_parser/include/tinyxml
    ${CMAKE_CURRENT_BINARY_DIR})
set(urdfdom_LIBRARIES urdfdom)

# libSGM
set(libSGM_VERSION_MAJOR 2)
set(libSGM_VERSION_MINOR 7)
set(libSGM_VERSION_PATCH 0)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libSGM/include/libsgm_config.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/libSGM/include/libsgm_config.h
)
file(GLOB STEREOSRCS "libSGM/src/*.cu" "libSGM/src/*.cpp")
cuda_add_library(sgm ${STEREOSRCS})
target_include_directories(sgm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libSGM/include)
target_link_libraries(sgm ${CUDA_LIBRARIES})
set(SGM_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/libSGM/include)
set(SGM_LIBRARIES sgm)

list(APPEND 3RDPARTY_INCLUDE_DIRS
     ${EIGEN3_INCLUDE_DIRS}
     ${flann_INCLUDE_DIRS}
     ${GLEW_INCLUDE_DIRS}
     ${GLFW_INCLUDE_DIRS}
     ${JPEG_TURBO_INCLUDE_DIRS}
     ${PNG_INCLUDE_DIRS}
     ${rply_INCLUDE_DIRS}
     ${JSONCPP_INCLUDE_DIRS}
     ${liblzf_INCLUDE_DIRS}
     ${tomasakeninemoeller_INCLUDE_DIRS}
     ${dlpack_INCLUDE_DIRS}
     ${rmm_INCLUDE_DIRS}
     ${imgui_INCLUDE_DIRS}
     ${stdgpu_INCLUDE_DIRS}
     ${spdlog_INCLUDE_DIRS}
     ${SGM_INCLUDE_DIRS}
     ${tinyobjloader_INCLUDE_DIRS}
     ${googletest_INCLUDE_DIRS}
     ${lbvh_INCLUDE_DIRS}
     ${lbvh_index_INCLUDE_DIRS}
     ${urdfdom_INCLUDE_DIR}
)

# set 3RDPARTY_LIBRARIES
list(APPEND 3RDPARTY_LIBRARIES
     ${FLANN_LIBRARIES}
     ${GLEW_LIBRARIES}
     ${GLFW_LIBRARIES}
     ${JPEG_TURBO_LIBRARIES}
     ${PNG_LIBRARIES}
     ${liblzf_LIBRARIES}
     ${rply_LIBRARIES}
     ${JSONCPP_LIBRARIES}
     ${stdgpu_LIBRARIES}
     ${SGM_LIBRARIES}
     ${tinyobjloader_LIBRARIES}
     ${googletest_LIBRARIES}
     ${imgui_LIBRARIES}
     ${urdfdom_LIBRARIES}
     ${CUDA_LIBRARIES}
)

set(3RDPARTY_INCLUDE_DIRS ${3RDPARTY_INCLUDE_DIRS} PARENT_SCOPE)
set(3RDPARTY_LIBRARIES ${3RDPARTY_LIBRARIES} PARENT_SCOPE)

install(TARGETS ${FLANN_LIBRARIES}
                ${GLEW_LIBRARIES}
                ${glfw_LIBRARIES}
                ${JPEG_TURBO_LIBRARIES}
                ${PNG_LIBRARIES}
                ${liblzf_LIBRARIES}
                ${rply_LIBRARIES}
                ${JSONCPP_LIBRARIES}
                ${SGM_LIBRARIES}
                ${tinyobjloader_LIBRARIES}
                ${googletest_LIBRARIES}
                ${imgui_LIBRARIES}
                ${urdfdom_LIBRARIES}
    EXPORT ${PROJECT_NAME}3rdTargets
    RUNTIME DESTINATION ${CUPOCH_INSTALL_BIN_DIR}
    LIBRARY DESTINATION ${CUPOCH_INSTALL_LIB_DIR}
    ARCHIVE DESTINATION ${CUPOCH_INSTALL_LIB_DIR}
)

install(DIRECTORY ${rmm_INCLUDE_DIRS}
                ${stdgpu_INCLUDE_DIRS}
                ${SGM_INCLUDE_DIRS}
    DESTINATION ${CUPOCH_INSTALL_INCLUDE_DIR}/../
)