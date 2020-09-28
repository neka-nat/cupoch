cmake_minimum_required(VERSION 3.16.0)

project(cupoch_camera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Check for dependencies
find_package(Eigen3 CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_path(THRUST_INCLUDE_DIRS "thrust/addressof.h")

# Define the library
file(GLOB_RECURSE ALL_CPP_SOURCE_FILES "*.cpp")

add_library(${PROJECT_NAME} STATIC
    ${ALL_CPP_SOURCE_FILES}
)

target_include_directories(${PROJECT_NAME}
    PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${THRUST_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen jsoncpp_lib cupoch_utility)

# Installation
# Add 'd' suffix to debug builds
set_property(TARGET ${PROJECT_NAME} PROPERTY DEBUG_POSTFIX d)

# Installation
include(GNUInstallDirs)

file(GLOB_RECURSE ALL_HEADER_FILES "*.h")

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${ALL_HEADER_FILES}")

export(TARGETS ${PROJECT_NAME} FILE "${PROJECT_NAME}Config.cmake")

install(TARGETS ${PROJECT_NAME}
    EXPORT "${PROJECT_NAME}Config"
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cupoch/camera
)

install(EXPORT "${PROJECT_NAME}Config"
    FILE "${PROJECT_NAME}Config.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/
)
