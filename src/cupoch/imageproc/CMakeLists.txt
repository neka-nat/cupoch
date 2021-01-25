file(GLOB_RECURSE IMAGEPROC_SOURCE_FILES "*.cpp")
cuda_add_library(cupoch_imageproc ${IMAGEPROC_SOURCE_FILES})
target_link_libraries(cupoch_imageproc cupoch_geometry
                      ${3RDPARTY_LIBRARIES})