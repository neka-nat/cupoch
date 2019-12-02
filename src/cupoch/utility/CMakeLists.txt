file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")
cuda_add_library(cupoch_utility ${ALL_SOURCE_FILES})
target_link_libraries(cupoch_utility fmt ${CUDA_LIBRARIES})