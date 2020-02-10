# build
file(GLOB_RECURSE ALL_CUDA_SOURCE_FILES "*.cu")

# create object library
cuda_add_library(cupoch_integration ${ALL_CUDA_SOURCE_FILES})
target_link_libraries(cupoch_integration cupoch_geometry)