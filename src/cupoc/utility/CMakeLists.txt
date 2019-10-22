file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")
cuda_add_library(cupoc_utility ${ALL_SOURCE_FILES})
target_link_libraries(cupoc_utility fmt)