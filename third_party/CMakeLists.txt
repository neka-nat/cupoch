find_package(pybind11  QUIET)

if (BUILD_PYBIND11)
    add_subdirectory(pybind11)
endif ()

if (BUILD_UNIT_TESTS)
    message(STATUS "Building googletest from source")
    include_directories("googletest/googletest/include")
    include_directories("googletest/googletest/")
    set(googletest_INCLUDE_DIRS
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googlemock/include
      ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googlemock/)
    set(googletest_SOURCE
        googletest/googletest/src/gtest-all.cc
        googletest/googlemock/src/gmock-all.cc)
    include_directories(${googletest_INCLUDE_DIRS})
    add_library(googletest STATIC ${googletest_SOURCE})
    if (NOT BUILD_SHARED_LIBS)
      install(TARGETS googletest
            RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
            ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    endif()
    set(googletest_LIBRARIES googletest)
endif()

if (BUILD_EIGEN3)
    set(EIGEN3_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/third_party/eigen")
endif ()

set(flann_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)

include_directories(fmt/include)
cuda_add_library(fmt fmt/src/format.cc fmt/src/posix.cc)
set(fmt_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/fmt/include)

list(APPEND 3RDPARTY_INCLUDE_DIRS
     ${EIGEN3_INCLUDE_DIRS}
     ${flann_INCLUDE_DIRS}
     ${fmt_INCLUDE_DIRS}
)

set(3RDPARTY_INCLUDE_DIRS ${3RDPARTY_INCLUDE_DIRS} PARENT_SCOPE)