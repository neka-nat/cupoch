configure_file("${PROJECT_SOURCE_DIR}/src/cupoch/cupoch_config.h.in"
               "${PROJECT_SOURCE_DIR}/src/cupoch/cupoch_config.h")

set(USE_RMM OFF)
add_definitions(-DWINDOWS)
add_definitions(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_WARNINGS)
add_definitions(-DGLEW_STATIC)		# enable GLEW for windows
add_definitions(-DTHRUST_CPP11_REQUIRED_NO_ERROR)
SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_definitions(-DNOMINMAX)
add_definitions(-D_USE_MATH_DEFINES)

# Make a list of CUDA architectures
option(SUPPORT_MAXWELL "Support Maxwell architectures" ON)
option(SUPPORT_PASCAL  "Support Pascal architectures"  ON)
option(SUPPORT_VOLTA   "Support Volta architectures"   ON)
option(SUPPORT_TURING  "Support Turing architectures"  ON)
option(SUPPORT_AMPERE  "Support Ampere architectures"  OFF)

if(SUPPORT_MAXWELL)
list(APPEND SUPPORTED_ARCHITECTURE 50 52 53)
endif()

if(SUPPORT_PASCAL)
list(APPEND SUPPORTED_ARCHITECTURE 60 61 62)
endif()

if(SUPPORT_VOLTA)
list(APPEND SUPPORTED_ARCHITECTURE 70)
endif()

if(SUPPORT_TURING)
list(APPEND SUPPORTED_ARCHITECTURE 75)
endif()

if(SUPPORT_AMPERE)
list(APPEND SUPPORTED_ARCHITECTURE 86)
endif()

add_subdirectory(utility)
add_subdirectory(camera)
add_subdirectory(geometry)
add_subdirectory(collision)
add_subdirectory(integration)
add_subdirectory(registration)
add_subdirectory(odometry)
add_subdirectory(planning)
# add_subdirectory(io)
# add_subdirectory(visualization)

add_library(cupoch INTERFACE)
target_link_libraries(cupoch 
    INTERFACE 
        cupoch_utility cupoch_camera cupoch_geometry cupoch_collision 
        cupoch_integration cupoch_registration cupoch_odometry cupoch_planning cupoch_io
    )

target_include_directories(cupoch
    INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Installation

include(GNUInstallDirs)

export(TARGETS cupoch NAMESPACE cupoch:: FILE cupochTargets.cmake)

install(TARGETS ${PROJECT_NAME}
    EXPORT cupochTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    cupochConfigVersion.cmake
    VERSION 0.1.3
    COMPATIBILITY SameMajorVersion
)

install(EXPORT cupochTargets
    FILE cupochTargets.cmake
    NAMESPACE cupoch::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/
)

install(FILES cmake/cupochConfig.cmake "${CMAKE_CURRENT_BINARY_DIR}/cupochConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/
)
