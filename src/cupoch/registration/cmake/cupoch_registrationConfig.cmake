include(CMakeFindDependencyMacro)
find_package(Eigen3 3.3.7 CONFIG REQUIRED)
find_package(stdgpu CONFIG REQUIRED)
include("${CMAKE_CURRENT_LIST_DIR}/cupoch_registrationTargets.cmake")