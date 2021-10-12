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

macro(EXAMPLE_CU EXAMPLE_CU_NAME)
    cuda_add_executable(${EXAMPLE_CU_NAME} "${EXAMPLE_CU_NAME}.cu")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${EXAMPLE_CU_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${EXAMPLE_CU_NAME} PROPERTIES
            FOLDER "examples/cpp/"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/examples")
endmacro(EXAMPLE_CU_NAME)

EXAMPLE_CPP(image cupoch_visualization cupoch_geometry cupoch_io ${3RDPARTY_LIBRARIES})
EXAMPLE_CPP(pointcloud cupoch_visualization cupoch_geometry cupoch_io ${3RDPARTY_LIBRARIES})
EXAMPLE_CPP(trianglemesh cupoch_visualization cupoch_geometry cupoch_io ${3RDPARTY_LIBRARIES})
EXAMPLE_CPP(registration cupoch_visualization cupoch_registration cupoch_io ${3RDPARTY_LIBRARIES})
EXAMPLE_CPP(voxelization cupoch_visualization cupoch_geometry cupoch_io ${3RDPARTY_LIBRARIES})
EXAMPLE_CPP(kinematics cupoch_visualization cupoch_kinematics cupoch_io ${3RDPARTY_LIBRARIES})
EXAMPLE_CU(trial cupoch_visualization cupoch_kinematics cupoch_io cupoch_registration ${3RDPARTY_LIBRARIES})