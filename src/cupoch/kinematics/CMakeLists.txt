file(GLOB_RECURSE KINEMATICS_SOURCE_FILES "*.cpp")

# Create object library
add_library(cupoch_kinematics ${KINEMATICS_SOURCE_FILES})
target_link_libraries(cupoch_kinematics cupoch_collision
                      cupoch_utility
                      ${3RDPARTY_LIBRARIES})