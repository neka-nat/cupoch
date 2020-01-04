file(GLOB_RECURSE CAMERA_SOURCE_FILES "*.cpp")
cuda_add_library(cupoch_camera ${CAMERA_SOURCE_FILES})