cmake_minimum_required(VERSION 3.16.0)

project(cupoch_collision VERSION 0.1.3 LANGUAGES CXX CUDA)

# Check for dependencies
find_package(Eigen3 3.3.7 CONFIG REQUIRED)
find_package(libvh CONFIG REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --no-host-device-move-forward --expt-relaxed-constexpr --extended-lambda")

# Define the library
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cu")
file(GLOB_RECURSE ALL_HEADER_FILES "*.h" "*.inl")

add_library(${PROJECT_NAME} STATIC
    ${ALL_SOURCE_FILES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES 
    CUDA_ARCHITECTURES "${SUPPORTED_ARCHITECTURE}"
    CXX_STANDARD 14
    CUDA_SEPARABLE_COMPILATION OFF
    DEBUG_POSTFIX "_d"
    PUBLIC_HEADER "${ALL_HEADER_FILES}"
)

target_include_directories(${PROJECT_NAME}
    PUBLIC 
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../>
)

target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen libvh cupoch_geometry)

# Installation

include(GNUInstallDirs)

export(TARGETS ${PROJECT_NAME} NAMESPACE cupoch:: FILE "${PROJECT_NAME}Targets.cmake")

install(TARGETS ${PROJECT_NAME}
    EXPORT "${PROJECT_NAME}Targets"
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cupoch/collision
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT "${PROJECT_NAME}Targets"
    FILE "${PROJECT_NAME}Targets.cmake"
    NAMESPACE cupoch::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/
)

install(FILES "cmake/${PROJECT_NAME}Config.cmake" "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/
)
