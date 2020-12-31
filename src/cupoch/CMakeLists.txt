configure_file("${PROJECT_SOURCE_DIR}/src/cupoch/cupoch_config.h.in"
               "${PROJECT_SOURCE_DIR}/src/cupoch/cupoch_config.h")

add_subdirectory(camera)
add_subdirectory(collision)
add_subdirectory(geometry)
add_subdirectory(integration)
add_subdirectory(io)
add_subdirectory(kinematics)
add_subdirectory(kinfu)
add_subdirectory(odometry)
add_subdirectory(planning)
add_subdirectory(registration)
add_subdirectory(utility)
add_subdirectory(visualization)

# Installation
install(TARGETS ${PROJECT_NAME}_camera
                ${PROJECT_NAME}_collision
                ${PROJECT_NAME}_geometry
                ${PROJECT_NAME}_integration
                ${PROJECT_NAME}_io
                ${PROJECT_NAME}_kinematics
                ${PROJECT_NAME}_kinfu
                ${PROJECT_NAME}_odometry
                ${PROJECT_NAME}_planning
                ${PROJECT_NAME}_registration
                ${PROJECT_NAME}_utility
                ${PROJECT_NAME}_visualization
        EXPORT ${PROJECT_NAME}Targets
        RUNTIME DESTINATION ${CUPOCH_INSTALL_BIN_DIR}
        LIBRARY DESTINATION ${CUPOCH_INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${CUPOCH_INSTALL_LIB_DIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${CUPOCH_INSTALL_INCLUDE_DIR}
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.inl"
)
