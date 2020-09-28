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

add_subdirectory(utility)
add_subdirectory(camera)
add_subdirectory(geometry)
add_subdirectory(collision)
add_subdirectory(integration)
add_subdirectory(registration)
add_subdirectory(odometry)
add_subdirectory(planning)
add_subdirectory(io)
# add_subdirectory(visualization)

