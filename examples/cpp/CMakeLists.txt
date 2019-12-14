macro(EXAMPLE_CPP EXAMPLE_CPP_NAME)
    add_executable(${EXAMPLE_CPP_NAME} "${EXAMPLE_CPP_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXAMPLE_CPP_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${EXAMPLE_CPP_NAME} PROPERTIES
            FOLDER "examples/cpp/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")
endmacro(EXAMPLE_CPP)

EXAMPLE_CPP(pointcloud cupoch_geometry cupoch_io)
EXAMPLE_CPP(registration cupoch_registration cupoch_io)