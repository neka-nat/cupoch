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

set(flann_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/flann/src/cpp)

file(GLOB_RECURSE C_SOURCES flann/src/cpp/flann/flann.cpp)
file(GLOB_RECURSE CPP_SOURCES flann/src/cpp/flann/flann_cpp.cpp)
file(GLOB_RECURSE CU_SOURCES flann/src/cpp/flann/*.cu)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DFLANN_USE_CUDA")
cuda_add_library(flann_cuda_s STATIC ${CU_SOURCES})
target_include_directories(flann_cuda_s PUBLIC ${flann_INCLUDE_DIRS})


if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_COMPILER_IS_GNUCC)
    cuda_add_library(flann_cuda SHARED "")
    set_target_properties(flann_cuda PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(flann_cuda -Wl,-whole-archive flann_cuda_s -Wl,-no-whole-archive)
else()
    cuda_add_library(flann_cuda SHARED ${CPP_SOURCES})
endif()


cuda_add_library(fmt fmt/src/format.cc fmt/src/posix.cc)
target_include_directories(fmt PUBLIC fmt/include)
set(fmt_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/fmt/include)

# liblzf
file(GLOB LIBLZF_SOURCE_FILES "liblzf/*.c")
add_library(liblzf ${LIBLZF_SOURCE_FILES})
target_include_directories(liblzf PUBLIC liblzf)
set(liblzf_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/liblzf)

# rply
add_library(rply rply/rply.c)
set(rply_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/rply)

list(APPEND 3RDPARTY_INCLUDE_DIRS
     ${EIGEN3_INCLUDE_DIRS}
     ${flann_INCLUDE_DIRS}
     ${rply_INCLUDE_DIRS}
     ${fmt_INCLUDE_DIRS}
     ${liblzf_INCLUDE_DIRS}
)

set(3RDPARTY_INCLUDE_DIRS ${3RDPARTY_INCLUDE_DIRS} PARENT_SCOPE)