cmake_minimum_required(VERSION 3.16.0)

project(cupoch_utility LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Check for dependencies
find_package(CUDA REQUIRED)
find_package(dlpack CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)

enable_language("CUDA")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
add_definitions(-DUSE_CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr;--extended-lambda")

# Define the library

file(GLOB_RECURSE ALL_CPP_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE ALL_CUDA_SOURCE_FILES "*.cu")

cuda_add_library(${PROJECT_NAME} STATIC
    ${ALL_CPP_SOURCE_FILES}
    ${ALL_CUDA_SOURCE_FILES}
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 35)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME}
    PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
)

target_link_libraries(${PROJECT_NAME} dlpack::dlpack Eigen3::Eigen fmt::fmt fmt::fmt-header-only jsoncpp_lib)

# Installation
# Add 'd' suffix to debug builds
set_property(TARGET ${PROJECT_NAME} PROPERTY DEBUG_POSTFIX d)

# Installation
include(GNUInstallDirs)

file(GLOB_RECURSE ALL_HEADER_FILES "*.h")
file(GLOB_RECURSE ALL_HEADER_FILES_MISC "*.inl")

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${ALL_HEADER_FILES};${ALL_HEADER_FILES_MISC}")

# install(DIRECTORY "include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cupoch/utility)

export(TARGETS ${PROJECT_NAME} FILE "${PROJECT_NAME}Config.cmake")

install(TARGETS ${PROJECT_NAME}
    EXPORT "${PROJECT_NAME}Config"
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cupoch/utility
)

install(EXPORT "${PROJECT_NAME}Config"
    FILE "${PROJECT_NAME}Config.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/
)
