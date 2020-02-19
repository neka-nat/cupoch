file(GLOB_RECURSE ALL_CPP_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE ALL_CUDA_SOURCE_FILES "*.cu")
cuda_add_library(cupoch_utility ${ALL_CUDA_SOURCE_FILES} ${ALL_CPP_SOURCE_FILES})
target_link_libraries(cupoch_utility ${3RDPARTY_LIBRARIES})