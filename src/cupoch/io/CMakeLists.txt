file(GLOB_RECURSE IO_CPP_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE IO_CUDA_SOURCE_FILES "*.cu")
set(IO_ALL_SOURCE_FILES ${IO_CPP_SOURCE_FILES} ${IO_CUDA_SOURCE_FILES})

# Create object library
cuda_add_library(cupoch_io ${IO_ALL_SOURCE_FILES})
target_link_libraries(cupoch_io cupoch_geometry
                      cupoch_utility liblzf rply
                      ${JPEG_TURBO_LIBRARIES}
                      ${PNG_LIBRARIES})