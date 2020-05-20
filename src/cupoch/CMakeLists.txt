configure_file("${PROJECT_SOURCE_DIR}/src/cupoch/cupoch_config.h.in"
               "${PROJECT_SOURCE_DIR}/src/cupoch/cupoch_config.h")

add_subdirectory(camera)
add_subdirectory(collision)
add_subdirectory(geometry)
add_subdirectory(integration)
add_subdirectory(io)
add_subdirectory(odometry)
add_subdirectory(planning)
add_subdirectory(registration)
add_subdirectory(utility)
add_subdirectory(visualization)

# Install headers
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
        PATTERN     "Visualization/Shader/GLSL" EXCLUDE
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.cu" EXCLUDE
        PATTERN     "*.in" EXCLUDE
        PATTERN     "*.txt" EXCLUDE)

file(GLOB HEADERS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS_ROOT} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME})
