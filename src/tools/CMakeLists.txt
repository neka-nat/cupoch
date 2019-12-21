macro(TOOL TOOL_NAME)
    add_executable(${TOOL_NAME} "${TOOL_NAME}.cpp")

    set(DEPENDENCIES "${ARGN}")
    foreach(DEPENDENCY IN LISTS DEPENDENCIES)
        target_link_libraries(${TOOL_NAME} ${DEPENDENCY})
    endforeach()

    set_target_properties(${TOOL_NAME} PROPERTIES FOLDER "tools")
endmacro(TOOL)

TOOL(encode_shader)