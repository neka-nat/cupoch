cmake_minimum_required(VERSION 3.16.0)

project(cupoch_registration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Check for dependencies
find_package(CUDA REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(stdgpu CONFIG REQUIRED)
find_path(THRUST_INCLUDE_DIRS "thrust/addressof.h")

enable_language("CUDA")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
add_definitions(-DUSE_CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    --no-host-device-move-forward
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --default-stream per-thread
    --use_fast_math
    -Xcudafe "--diag_suppress=integer_sign_change"
    -Xcudafe "--diag_suppress=partial_override"
    -Xcudafe "--diag_suppress=virtual_function_decl_hidden"
)

# Define the library
file(GLOB_RECURSE ALL_CUDA_SOURCE_FILES "*.cu")

cuda_add_library(${PROJECT_NAME} STATIC
    ${ALL_CUDA_SOURCE_FILES}
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 35)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME}
    PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
)

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen stdgpu thrust cupoch_utility cupoch_geometry)

# Installation
# Add 'd' suffix to debug builds
set_property(TARGET ${PROJECT_NAME} PROPERTY DEBUG_POSTFIX d)

# Installation
include(GNUInstallDirs)

file(GLOB_RECURSE ALL_HEADER_FILES "*.h" "*.inl")

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${ALL_HEADER_FILES}")

export(TARGETS ${PROJECT_NAME} FILE "${PROJECT_NAME}Config.cmake")

install(TARGETS ${PROJECT_NAME}
    EXPORT "${PROJECT_NAME}Config"
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cupoch/registration
)

install(EXPORT "${PROJECT_NAME}Config"
    FILE "${PROJECT_NAME}Config.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/
)
