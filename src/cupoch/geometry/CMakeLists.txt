file(GLOB_RECURSE GEOMETRY_SOURCE_FILES "*.cu")
cuda_add_library(cupoch_geometry ${GEOMETRY_SOURCE_FILES})
target_link_libraries(cupoch_geometry cupoch_knn
                      cupoch_camera
                      ${3RDPARTY_LIBRARIES})