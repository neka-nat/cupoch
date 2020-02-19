file(GLOB_RECURSE GEOMETRY_SOURCE_FILES "*.cu")
cuda_add_library(cupoch_geometry ${GEOMETRY_SOURCE_FILES})
target_link_libraries(cupoch_geometry cupoch_utility
                      ${3RDPARTY_LIBRARIES})