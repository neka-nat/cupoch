cmake_minimum_required(VERSION 3.0)
file(GLOB_RECURSE UNIT_TESTS "*.cpp")

add_executable(unittests ${UNIT_TESTS})
add_definitions(-DTEST_DATA_DIR="${PROJECT_SOURCE_DIR}/examples/testdata")
target_link_libraries(unittests
    cupoch_registration cupoch_integration
    cupoch_io cupoch_camera cupoch_planning
    cupoch_utility googletest
    ${3RDPARTY_LIBRARIES})