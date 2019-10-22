cmake_minimum_required(VERSION 3.0)
file(GLOB_RECURSE CUDA_TESTS "*.cu")
file(GLOB_RECURSE UNIT_TESTS "*.cpp")

cuda_add_library(cuda_tests ${CUDA_TESTS})
add_executable(unittests ${UNIT_TESTS})
target_link_libraries(unittests cuda_tests cupoc_geometry cupoc_utility googletest pthread)