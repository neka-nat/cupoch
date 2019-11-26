cmake_minimum_required(VERSION 3.0)
file(GLOB_RECURSE UNIT_TESTS "*.cpp")

add_executable(unittests ${UNIT_TESTS})
target_link_libraries(unittests cupoc_geometry cupoc_utility googletest pthread)