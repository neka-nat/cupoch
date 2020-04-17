file(GLOB_RECURSE COLLISION_SOURCE_FILES "*.cu")
cuda_add_library(cupoch_collision ${COLLISION_SOURCE_FILES})
target_link_libraries(cupoch_collision cupoch_geometry
                      ${3RDPARTY_LIBRARIES})