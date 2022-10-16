file(GLOB SHADER_FILES "shader/glsl/*.glsl")

add_custom_command(
    OUTPUT ${PROJECT_SOURCE_DIR}/src/cupoch/visualization/shader/shader.h
    COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/encode_shader.py shader.h
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/cupoch/visualization/shader/
    DEPENDS ${SHADER_FILES}
)

foreach(file ${SHADER_FILES})
    add_custom_command(
        OUTPUT ${PROJECT_SOURCE_DIR}/src/cupoch/visualization/shader/shader.h
        COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/encode_shader.py shader.h ${file}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/cupoch/visualization/shader/
        DEPENDS ${SHADER_FILES}
        APPEND
    )
endforeach(file)

add_custom_target(shader_file_target ALL
                  DEPENDS ${PROJECT_SOURCE_DIR}/src/cupoch/visualization/shader/shader.h)
set_target_properties(shader_file_target PROPERTIES FOLDER "custom")

file(GLOB_RECURSE VISUALIZATION_CPP_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE VISUALIZATION_CUDA_SOURCE_FILES "*.cu")

# create object library
cuda_add_library(cupoch_visualization ${VISUALIZATION_CUDA_SOURCE_FILES} ${VISUALIZATION_CPP_SOURCE_FILES})
target_link_libraries(cupoch_visualization cupoch_geometry cupoch_io cupoch_camera ${3RDPARTY_LIBRARIES})
add_dependencies(cupoch_visualization shader_file_target)